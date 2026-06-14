//! Full-screen ratatui front end for `camelid chat`.
//!
//! The redraw loop runs on the main thread; each generation streams on a
//! background thread that forwards deltas over a channel, so the UI stays
//! responsive and Ctrl-C cancels mid-stream. Everything rides the same audited
//! HTTP/SSE client and the shared [`Session`] core.

use std::io::Stdout;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};
use ratatui::{Frame, Terminal};

use super::banner;
use super::client::StreamEnd;
use super::models::{Availability, PickerRow};
use super::session::{self, Role, Session};

const TAN: Color = Color::Indexed(179);
const SAND: Color = Color::Indexed(223);
const DIMC: Color = Color::Indexed(245);
const USERC: Color = Color::Indexed(110);
const CODEC: Color = Color::Indexed(151);

/// A message from the streaming worker thread to the UI loop.
enum StreamMsg {
    Delta(String),
    Done(u32),
    Cancelled,
    Error(String),
}

enum Mode {
    Normal,
    Picker,
    Help,
}

/// Work that must run with the terminal temporarily restored (so `curl`'s
/// download progress is visible): leave the alt-screen, run it, re-enter.
enum Suspend {
    Pull(String),
    DownloadAndLoad(usize),
}

struct PickerUi {
    rows: Vec<PickerRow>,
    state: ListState,
}

pub fn run(session: &mut Session, addr: SocketAddr, spawned: bool) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    let mut app = App::new(session, addr, spawned);
    let result = app.run(&mut terminal);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

struct App<'a> {
    session: &'a mut Session,
    addr: SocketAddr,
    spawned: bool,
    input: String,
    cursor: usize, // byte offset into input
    scroll: usize,
    last_max_scroll: usize,
    follow: bool,
    sidebar: bool,
    mode: Mode,
    picker: PickerUi,
    status: String,
    stream_rx: Option<Receiver<StreamMsg>>,
    streaming: bool,
    live: String,
    started: Option<Instant>,
    stats: Option<String>,
    history: Vec<String>,
    hist_idx: Option<usize>,
    pending: Option<Suspend>,
    quit: bool,
}

impl<'a> App<'a> {
    fn new(session: &'a mut Session, addr: SocketAddr, spawned: bool) -> Self {
        let status = format!(
            "server {} at {addr} · F1 help · Tab sidebar · /models to switch",
            if spawned { "spawned" } else { "attached" }
        );
        App {
            session,
            addr,
            spawned,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            last_max_scroll: 0,
            follow: true,
            sidebar: true,
            mode: Mode::Normal,
            picker: PickerUi {
                rows: Vec::new(),
                state: ListState::default(),
            },
            status,
            stream_rx: None,
            streaming: false,
            live: String::new(),
            started: None,
            stats: None,
            history: Vec::new(),
            hist_idx: None,
            pending: None,
            quit: false,
        }
    }

    fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
        if !self.session.has_model() {
            self.open_picker();
        }
        while !self.quit {
            terminal.draw(|f| self.draw(f))?;
            self.poll_stream();
            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(key),
                    Event::Mouse(m) => match m.kind {
                        MouseEventKind::ScrollUp => self.scroll_up(3),
                        MouseEventKind::ScrollDown => self.scroll_down(3),
                        _ => {}
                    },
                    _ => {}
                }
            }
            if let Some(op) = self.pending.take() {
                self.run_suspended(terminal, op)?;
            }
        }
        Ok(())
    }

    // ---- streaming -------------------------------------------------------

    fn start_generation(&mut self) {
        let request = self.session.build_request(true);
        session::CANCEL.store(false, Ordering::SeqCst);
        let client = self.session.client();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let forward = tx.clone();
            let result = client.chat_stream(&request, &session::CANCEL, move |delta| {
                let _ = forward.send(StreamMsg::Delta(delta.to_string()));
            });
            let _ = match result {
                Ok((StreamEnd::Done, n)) => tx.send(StreamMsg::Done(n)),
                Ok((StreamEnd::Cancelled, _)) => tx.send(StreamMsg::Cancelled),
                Err(err) => tx.send(StreamMsg::Error(err.to_string())),
            };
        });
        self.stream_rx = Some(rx);
        self.streaming = true;
        self.live.clear();
        self.started = Some(Instant::now());
        self.stats = None;
        self.follow = true;
    }

    fn poll_stream(&mut self) {
        if self.stream_rx.is_none() {
            return;
        }
        loop {
            let msg = match self.stream_rx.as_ref().unwrap().try_recv() {
                Ok(msg) => msg,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.finish(None);
                    return;
                }
            };
            match msg {
                StreamMsg::Delta(d) => {
                    self.live.push_str(&d);
                    if self.follow {
                        self.scroll = usize::MAX;
                    }
                }
                StreamMsg::Done(n) => {
                    self.finish(Some(n));
                    return;
                }
                StreamMsg::Cancelled => {
                    self.session.pop_last();
                    self.streaming = false;
                    self.stream_rx = None;
                    self.live.clear();
                    self.status = "interrupted — turn discarded".into();
                    return;
                }
                StreamMsg::Error(e) => {
                    self.session.pop_last();
                    self.streaming = false;
                    self.stream_rx = None;
                    self.live.clear();
                    self.status = format!("generation failed: {e}");
                    return;
                }
            }
        }
    }

    fn finish(&mut self, completion: Option<u32>) {
        let text = std::mem::take(&mut self.live);
        let secs = self
            .started
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        self.session.push_assistant(text);
        self.session.last_prompt_tokens = None;
        self.session.last_completion_tokens = completion;
        self.stats = Some(match completion {
            Some(n) if secs > 0.0 => format!("{n} tok · {:.0} tok/s · {secs:.1}s", n as f64 / secs),
            Some(n) => format!("{n} tok · {secs:.1}s"),
            None => format!("{secs:.1}s"),
        });
        self.streaming = false;
        self.stream_rx = None;
    }

    // ---- input / commands ------------------------------------------------

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.history.push(text.clone());
        self.hist_idx = None;
        self.input.clear();
        self.cursor = 0;
        if let Some(command) = text.strip_prefix('/') {
            self.command(command);
        } else if self.session.has_model() {
            self.session.push_user(text);
            self.start_generation();
        } else {
            self.status = "no model loaded — /models to pick one".into();
        }
    }

    fn command(&mut self, command: &str) {
        let mut parts = command.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "models" => self.open_picker(),
            "model" => self.select_by_id(arg),
            "set" => {
                let mut kv = arg.splitn(2, char::is_whitespace);
                match (kv.next(), kv.next()) {
                    (Some(k), Some(v)) if !k.is_empty() && !v.trim().is_empty() => {
                        self.status = match self.session.set_param(k, v.trim()) {
                            Ok(msg) => format!("✓ {msg}"),
                            Err(err) => format!("✗ {err}"),
                        };
                    }
                    _ => {
                        self.status = format!(
                            "usage: /set <name> <value> · {}",
                            self.session.settings.summary()
                        )
                    }
                }
            }
            "system" => {
                if arg.is_empty() {
                    self.status = match &self.session.system {
                        Some(t) => format!("system: {t}"),
                        None => "no system prompt set".into(),
                    };
                } else {
                    self.session.system = Some(arg.to_string());
                    self.status = "system prompt set (next turn)".into();
                }
            }
            "reset" | "clear" => {
                self.session.reset_history();
                self.status = "history cleared".into();
            }
            "retry" | "regenerate" => self.retry(),
            "tokens" => {
                let f = |v: Option<u32>| v.map(|n| n.to_string()).unwrap_or_else(|| "n/a".into());
                self.status = format!(
                    "last turn — prompt: {}  completion: {}",
                    f(self.session.last_prompt_tokens),
                    f(self.session.last_completion_tokens)
                );
            }
            "info" => {
                self.sidebar = true;
                self.status = "model info in the sidebar".into();
            }
            "save" => {
                let path = std::path::PathBuf::from(if arg.is_empty() {
                    "camelid-session.json"
                } else {
                    arg
                });
                self.status = match self.session.save(&path) {
                    Ok(()) => format!("saved → {}", path.display()),
                    Err(err) => format!("save failed: {err}"),
                };
            }
            "load" => {
                if arg.is_empty() {
                    self.status = "usage: /load <path.json>".into();
                } else {
                    self.status = match self.session.load(&std::path::PathBuf::from(arg)) {
                        Ok(()) => format!("loaded {} turn(s)", self.session.history.len()),
                        Err(err) => format!("load failed: {err}"),
                    };
                }
            }
            "pull" => {
                if arg.is_empty() {
                    self.status = "usage: /pull <alias>".into();
                } else {
                    self.pending = Some(Suspend::Pull(arg.to_string()));
                }
            }
            "help" => self.mode = Mode::Help,
            "exit" | "quit" => self.quit = true,
            other => self.status = format!("unknown command /{other} — /help"),
        }
    }

    fn retry(&mut self) {
        if !self.session.has_model() || self.session.last_user_message().is_none() {
            self.status = "nothing to retry".into();
            return;
        }
        if matches!(
            self.session.history.last().map(|t| t.role),
            Some(Role::Assistant)
        ) {
            self.session.pop_last();
        }
        self.start_generation();
    }

    // ---- model picker ----------------------------------------------------

    fn open_picker(&mut self) {
        let rows = self.session.supported_rows();
        self.picker
            .state
            .select(if rows.is_empty() { None } else { Some(0) });
        self.picker.rows = rows;
        self.mode = Mode::Picker;
    }

    fn picker_move(&mut self, delta: isize) {
        let len = self.picker.rows.len();
        if len == 0 {
            return;
        }
        let cur = self.picker.state.selected().unwrap_or(0) as isize;
        let next = (cur + delta).rem_euclid(len as isize) as usize;
        self.picker.state.select(Some(next));
    }

    fn picker_choose(&mut self) {
        let Some(index) = self.picker.state.selected() else {
            return;
        };
        let Some(row) = self.picker.rows.get(index) else {
            return;
        };
        match row.availability {
            Availability::Ready => {
                let id = row.id.clone();
                let path = row.local_path(self.session.models_dir());
                self.mode = Mode::Normal;
                if let Some(path) = path {
                    self.load_path(&path, &id);
                }
            }
            Availability::NotDownloaded => {
                self.mode = Mode::Normal;
                self.pending = Some(Suspend::DownloadAndLoad(index));
            }
            Availability::NoPullAlias => {
                self.status = format!("'{}' has no pull alias — use --model", row.id);
            }
        }
    }

    fn select_by_id(&mut self, id: &str) {
        let rows = self.session.supported_rows();
        let Some(row) = rows.iter().find(|r| r.id == id) else {
            self.status = format!("'{id}' is not a supported id — /models");
            return;
        };
        match row.availability {
            Availability::Ready => {
                if let Some(path) = row.local_path(self.session.models_dir()) {
                    let id = row.id.clone();
                    self.load_path(&path, &id);
                }
            }
            Availability::NotDownloaded => {
                self.status = format!("'{id}' not downloaded — /models to fetch")
            }
            Availability::NoPullAlias => self.status = format!("'{id}' has no pull alias"),
        }
    }

    fn load_path(&mut self, path: &std::path::Path, label: &str) {
        match self
            .session
            .load_model_file(path, Some(label), Some("supported"))
        {
            Ok(session::LoadResult::Loaded) => {
                self.status = format!("active model: {label} (history reset)")
            }
            Ok(session::LoadResult::Unsupported(message)) => self.status = message,
            Err(err) => self.status = format!("load failed: {err}"),
        }
    }

    /// Restore the terminal, run a blocking download/load with visible progress,
    /// then re-enter the alt-screen.
    fn run_suspended(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
        op: Suspend,
    ) -> anyhow::Result<()> {
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;

        match op {
            Suspend::Pull(alias) => {
                if let Err(err) =
                    camelid::catalog::run_pull(Some(&alias), self.session.models_dir())
                {
                    self.status = format!("pull failed: {err}");
                } else {
                    self.status = "downloaded — /models to load it".into();
                }
            }
            Suspend::DownloadAndLoad(index) => {
                let rows = self.session.supported_rows();
                if let Some(row) = rows.get(index) {
                    if let Some(item) = row.catalog.as_ref() {
                        match camelid::catalog::run_pull(
                            Some(item.catalog_id),
                            self.session.models_dir(),
                        ) {
                            Ok(()) => {
                                if let Some(path) = row.local_path(self.session.models_dir()) {
                                    let id = row.id.clone();
                                    self.load_path(&path, &id);
                                }
                            }
                            Err(err) => self.status = format!("pull failed: {err}"),
                        }
                    }
                }
            }
        }

        enable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            EnterAlternateScreen,
            EnableMouseCapture
        )?;
        terminal.clear()?;
        Ok(())
    }

    // ---- scrolling -------------------------------------------------------

    fn scroll_up(&mut self, n: usize) {
        // Resolve a pending "stick to bottom" to a concrete value first.
        if self.scroll == usize::MAX {
            self.scroll = self.last_max_scroll;
        }
        self.scroll = self.scroll.saturating_sub(n);
        self.follow = false;
    }

    fn scroll_down(&mut self, n: usize) {
        if self.scroll == usize::MAX {
            return;
        }
        self.scroll = (self.scroll + n).min(self.last_max_scroll);
        if self.scroll >= self.last_max_scroll {
            self.follow = true;
        }
    }

    // ---- key handling ----------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        // Ctrl-D quits from any mode (overlay or not).
        if ctrl && key.code == KeyCode::Char('d') {
            self.quit = true;
            return;
        }

        match self.mode {
            Mode::Help => {
                self.mode = Mode::Normal;
                return;
            }
            Mode::Picker => {
                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => self.picker_move(-1),
                    KeyCode::Down | KeyCode::Char('j') => self.picker_move(1),
                    KeyCode::Enter => self.picker_choose(),
                    KeyCode::Esc | KeyCode::Char('q') => self.mode = Mode::Normal,
                    _ => {}
                }
                return;
            }
            Mode::Normal => {}
        }

        match key.code {
            KeyCode::Char('d') if ctrl => self.quit = true,
            KeyCode::Char('c') if ctrl => {
                if self.streaming {
                    session::CANCEL.store(true, Ordering::SeqCst);
                } else if !self.input.is_empty() {
                    self.input.clear();
                    self.cursor = 0;
                } else {
                    self.status = "Ctrl-D or /exit to quit".into();
                }
            }
            KeyCode::Char('l') if ctrl => {
                self.session.reset_history();
                self.status = "history cleared".into();
            }
            KeyCode::Tab => self.sidebar = !self.sidebar,
            KeyCode::F(1) => self.mode = Mode::Help,
            KeyCode::PageUp => self.scroll_up(10),
            KeyCode::PageDown => self.scroll_down(10),
            KeyCode::Enter if alt => self.insert_char('\n'),
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => self.backspace(),
            KeyCode::Left => self.move_cursor(-1),
            KeyCode::Right => self.move_cursor(1),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            KeyCode::Char(c) if !ctrl => self.insert_char(c),
            _ => {}
        }
    }

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .next_back()
            .map(char::len_utf8)
            .unwrap_or(1);
        self.cursor -= prev;
        self.input.remove(self.cursor);
    }

    fn move_cursor(&mut self, delta: isize) {
        if delta < 0 {
            if let Some(c) = self.input[..self.cursor].chars().next_back() {
                self.cursor -= c.len_utf8();
            }
        } else if let Some(c) = self.input[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            Some(0) => 0,
            Some(i) => i - 1,
            None => self.history.len() - 1,
        };
        self.hist_idx = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor = self.input.len();
    }

    fn history_next(&mut self) {
        match self.hist_idx {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                self.input = self.history[i + 1].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.hist_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
            None => {}
        }
    }

    // ---- rendering -------------------------------------------------------

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let body = if self.sidebar {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(20), Constraint::Length(34)])
                .split(area);
            self.draw_sidebar(f, cols[1]);
            cols[0]
        } else {
            area
        };

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3), // chat
                Constraint::Length(self.input_height()),
                Constraint::Length(1), // status
            ])
            .split(body);

        self.draw_chat(f, rows[0]);
        self.draw_input(f, rows[1]);
        self.draw_status(f, rows[2]);

        match self.mode {
            Mode::Picker => self.draw_picker(f, area),
            Mode::Help => draw_help(f, area),
            Mode::Normal => {}
        }
    }

    fn input_height(&self) -> u16 {
        let lines = self.input.split('\n').count().max(1) as u16;
        (lines + 2).min(8)
    }

    fn draw_chat(&mut self, f: &mut Frame, area: Rect) {
        let title = if self.session.has_model() {
            format!(
                " 🐪 Camelid — {} · {} ",
                self.session.active_label, self.session.active_posture
            )
        } else {
            " 🐪 Camelid — no model ".to_string()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TAN))
            .title(Span::styled(
                title,
                Style::default().fg(SAND).add_modifier(Modifier::BOLD),
            ))
            .title_alignment(Alignment::Left);
        let inner = block.inner(area);
        f.render_widget(block, area);

        let width = inner.width.max(1) as usize;
        let lines = self.chat_lines(width);
        let height = inner.height as usize;
        let max_scroll = lines.len().saturating_sub(height);
        self.last_max_scroll = max_scroll;
        let start = if self.scroll == usize::MAX || self.follow {
            max_scroll
        } else {
            self.scroll.min(max_scroll)
        };
        let visible: Vec<Line> = lines.into_iter().skip(start).take(height).collect();
        f.render_widget(Paragraph::new(visible), inner);
    }

    /// Build the wrapped, styled transcript (history + any in-flight reply).
    fn chat_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut out: Vec<Line<'static>> = Vec::new();
        if self.session.history.is_empty() && !self.streaming {
            for line in banner::CAMEL_LINES.lines() {
                out.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(TAN),
                )));
            }
            out.push(Line::from(""));
            out.push(Line::from(Span::styled(
                "Type a message and press Enter. /help for commands.".to_string(),
                Style::default().fg(DIMC),
            )));
            return out;
        }
        for turn in &self.session.history {
            push_turn(&mut out, turn.role, &turn.content, width);
        }
        if self.streaming {
            push_turn(&mut out, Role::Assistant, &self.live, width);
            out.push(Line::from(Span::styled(
                "▌".to_string(),
                Style::default().fg(TAN),
            )));
        }
        out
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let border = if self.streaming { DIMC } else { TAN };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .title(Span::styled(" › ", Style::default().fg(SAND)));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let text: Vec<Line> = self
            .input
            .split('\n')
            .map(|l| Line::from(l.to_string()))
            .collect();
        let (row, col) = self.cursor_rowcol();
        f.render_widget(Paragraph::new(text), inner);
        if !self.streaming {
            f.set_cursor_position((inner.x + col as u16, inner.y + row as u16));
        }
    }

    fn cursor_rowcol(&self) -> (usize, usize) {
        let before = &self.input[..self.cursor];
        let row = before.matches('\n').count();
        let col = before
            .rsplit('\n')
            .next()
            .map(|s| s.chars().count())
            .unwrap_or(0);
        (row, col)
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let mut spans = Vec::new();
        if self.streaming {
            spans.push(Span::styled("● streaming ", Style::default().fg(TAN)));
            spans.push(Span::styled("^C cancel  ", Style::default().fg(DIMC)));
        } else if let Some(stats) = &self.stats {
            spans.push(Span::styled(
                format!("└ {stats}  "),
                Style::default().fg(DIMC),
            ));
        }
        spans.push(Span::styled(self.status.clone(), Style::default().fg(DIMC)));
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_sidebar(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DIMC))
            .title(Span::styled(" settings ", Style::default().fg(SAND)));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let s = &self.session.settings;
        let mut lines = vec![
            kv("model", &self.session.active_label),
            kv("posture", &self.session.active_posture),
            Line::from(""),
            kv("temperature", &format!("{:.2}", s.temperature)),
            kv("top_p", &opt(s.top_p.map(|v| format!("{v:.2}")))),
            kv("top_k", &opt(s.top_k.map(|v| v.to_string()))),
            kv("max_tokens", &s.max_tokens.to_string()),
            kv("seed", &opt(s.seed.map(|v| v.to_string()))),
            kv("stream", if s.stream { "on" } else { "off" }),
            Line::from(""),
            kv(
                "system",
                if self.session.system.is_some() {
                    "set"
                } else {
                    "—"
                },
            ),
            kv("turns", &self.session.history.len().to_string()),
            kv("server", &self.addr.to_string()),
            kv("via", if self.spawned { "spawned" } else { "attached" }),
        ];
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Tab hide · /set k v".to_string(),
            Style::default().fg(DIMC),
        )));
        f.render_widget(Paragraph::new(lines), inner.inner(Margin::new(1, 0)));
    }

    fn draw_picker(&mut self, f: &mut Frame, area: Rect) {
        let rect = centered(area, 64, 60);
        f.render_widget(Clear, rect);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TAN))
            .title(Span::styled(
                " supported models — ↑↓ Enter · Esc ",
                Style::default().fg(SAND).add_modifier(Modifier::BOLD),
            ));
        if self.picker.rows.is_empty() {
            let p = Paragraph::new("the server advertises no supported models")
                .block(block)
                .style(Style::default().fg(DIMC));
            f.render_widget(p, rect);
            return;
        }
        let items: Vec<ListItem> = self
            .picker
            .rows
            .iter()
            .map(|row| {
                let tag = match row.availability {
                    Availability::Ready => Span::styled(" [ready]", Style::default().fg(CODEC)),
                    Availability::NotDownloaded => {
                        Span::styled(" [not downloaded]", Style::default().fg(DIMC))
                    }
                    Availability::NoPullAlias => {
                        Span::styled(" [no pull alias]", Style::default().fg(DIMC))
                    }
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<32} ", row.id), Style::default().fg(SAND)),
                    Span::styled(format!("{:<6}", row.quant), Style::default().fg(DIMC)),
                    tag,
                ]))
            })
            .collect();
        let list = List::new(items)
            .block(block)
            .highlight_style(
                Style::default()
                    .bg(Color::Indexed(238))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› ");
        f.render_stateful_widget(list, rect, &mut self.picker.state);
    }
}

fn push_turn(out: &mut Vec<Line<'static>>, role: Role, content: &str, width: usize) {
    let color = match role {
        Role::User => USERC,
        Role::Assistant => TAN,
    };
    out.push(Line::from(Span::styled(
        format!("▸ {}", role.display()),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )));
    let mut in_code = false;
    for raw in content.split('\n') {
        if raw.trim_start().starts_with("```") {
            in_code = !in_code;
            out.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(DIMC),
            )));
            continue;
        }
        let style = if in_code {
            Style::default().fg(CODEC).bg(Color::Indexed(236))
        } else {
            Style::default()
        };
        for piece in wrap(raw, width.max(1)) {
            out.push(Line::from(Span::styled(piece, style)));
        }
    }
    out.push(Line::from(""));
}

fn draw_help(f: &mut Frame, area: Rect) {
    let rect = centered(area, 62, 70);
    f.render_widget(Clear, rect);
    let lines: Vec<Line> = [
        ("Enter", "send · Alt+Enter newline"),
        ("Ctrl-C", "cancel a stream / clear input"),
        ("Ctrl-D", "quit · Ctrl-L clear history"),
        ("PgUp/PgDn", "scroll · wheel scrolls too"),
        ("Tab", "toggle the settings sidebar"),
        ("Up/Down", "input history"),
        ("", ""),
        ("/models", "pick a supported model"),
        ("/model <id>", "switch model by id"),
        ("/set k v", "temperature top_p top_k max_tokens seed stream"),
        ("/system <t>", "system prompt · /reset clear · /retry"),
        ("/save /load", "session JSON · /pull <alias> download"),
        ("/tokens /info", "stats · /help · /exit"),
    ]
    .iter()
    .map(|(k, v)| {
        Line::from(vec![
            Span::styled(format!("  {k:<12}"), Style::default().fg(SAND)),
            Span::styled((*v).to_string(), Style::default().fg(DIMC)),
        ])
    })
    .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(TAN))
        .title(Span::styled(
            " keys & commands — any key to close ",
            Style::default().fg(SAND).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(Paragraph::new(lines).block(block), rect);
}

fn kv(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key:<12}"), Style::default().fg(DIMC)),
        Span::styled(value.to_string(), Style::default().fg(SAND)),
    ])
}

fn opt(value: Option<String>) -> String {
    value.unwrap_or_else(|| "off".into())
}

fn centered(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_h) / 2),
            Constraint::Percentage(pct_h),
            Constraint::Percentage((100 - pct_h) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_w) / 2),
            Constraint::Percentage(pct_w),
            Constraint::Percentage((100 - pct_w) / 2),
        ])
        .split(v[1])[1]
}

/// Char-safe word wrap (no panics on multibyte; hard-splits over-long words).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    let flush_word = |out: &mut Vec<String>, cur: &mut String, cur_len: &mut usize, word: &str| {
        let wlen = word.chars().count();
        if wlen > width {
            // Hard-split a word longer than the line.
            let mut chunk = String::new();
            let mut clen = 0;
            for ch in word.chars() {
                if clen == width {
                    out.push(std::mem::take(&mut chunk));
                    clen = 0;
                }
                chunk.push(ch);
                clen += 1;
            }
            *cur = chunk;
            *cur_len = clen;
        } else {
            *cur = word.to_string();
            *cur_len = wlen;
        }
    };
    for word in text.split(' ') {
        let wlen = word.chars().count();
        if cur_len == 0 {
            flush_word(&mut out, &mut cur, &mut cur_len, word);
        } else if cur_len + 1 + wlen <= width {
            cur.push(' ');
            cur.push_str(word);
            cur_len += 1 + wlen;
        } else {
            out.push(std::mem::take(&mut cur));
            cur_len = 0;
            flush_word(&mut out, &mut cur, &mut cur_len, word);
        }
    }
    out.push(cur);
    out
}
