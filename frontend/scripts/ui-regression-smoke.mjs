#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  NEW_CHAT_SENTINEL,
  resolveSelectedConversation,
  shouldCreateConversationForSend,
} from '../src/lib/chatState.js'

const oldChat = { id: 'old-chat', title: 'Old chat', messages: [{ role: 'user', content: 'old prompt' }] }
const newerChat = { id: 'newer-chat', title: 'Newer chat', messages: [{ role: 'user', content: 'newer prompt' }] }
const conversations = [newerChat, oldChat]

assert.equal(resolveSelectedConversation(conversations, NEW_CHAT_SENTINEL), null, 'new-chat sentinel must render an empty landing, not the newest old chat')
assert.equal(resolveSelectedConversation(conversations, null), null, 'null selection must not silently fall back to an old chat')
assert.equal(resolveSelectedConversation(conversations, 'missing-chat'), null, 'missing selection must not silently fall back to an old chat')
assert.equal(resolveSelectedConversation(conversations, 'old-chat'), oldChat, 'explicit old-chat selection should still open that chat')
assert.equal(shouldCreateConversationForSend(null, NEW_CHAT_SENTINEL), true, 'sending from new-chat landing should create a fresh conversation')
assert.equal(shouldCreateConversationForSend(oldChat, NEW_CHAT_SENTINEL), true, 'the sentinel must win even if a stale selectedConversation prop exists')
assert.equal(shouldCreateConversationForSend(oldChat, 'old-chat'), false, 'sending from an explicit existing chat should append to that chat')

const readmeSource = readFileSync(new URL('../../README.md', import.meta.url), 'utf8')
assert.match(readmeSource, /docs\/assets\/camelid-readme-chat-surface-dark\.png/, 'README should use the approved dark collapsed-rail chat screenshot')
assert.doesNotMatch(readmeSource, /docs\/assets\/ui-screenshot-v2\.png/, 'README must not regress to the retired light screenshot')
assert.match(readmeSource, /dark, collapsed-rail chat surface/i, 'README caption should preserve the dark screenshot contract')

const chatWorkspaceSource = readFileSync(new URL('../src/views/ChatWorkspace.jsx', import.meta.url), 'utf8')
assert.match(chatWorkspaceSource, /pending is-streaming/, 'pending assistant row should use the same streaming Pac-Man state as live token rows')
assert.match(chatWorkspaceSource, /splitFenceInfo/, 'streaming/incomplete fenced code blocks should be parsed as code instead of prose')
assert.match(chatWorkspaceSource, /pushCodeBlock/, 'code block rendering should stay centralized for complete and incomplete fences')

const componentCss = readFileSync(new URL('../src/styles/components.css', import.meta.url), 'utf8')
assert.match(componentCss, /assistant\.is-streaming::before\s*{[^}]*camelid-pacman-chomp/s, 'Pac-Man should chomp while streaming')
assert.match(componentCss, /assistant\.is-streaming::after\s*{[^}]*camelid-pellets-feed/s, 'pellets should only appear on streaming assistant rows')
assert.doesNotMatch(componentCss, /camelid-pacman-bob/, 'Pac-Man should stay game-steady instead of bobbing')
assert.doesNotMatch(componentCss, /assistant::after\s*{/, 'completed assistant rows must not keep pellet animation')

console.log('UI regression smoke passed')
