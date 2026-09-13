export const normalizeCodeLanguage = (value) => {
  const language = String(value || '').trim().replace(/[^a-zA-Z0-9_+#.-].*$/, '')
  if (!language) return 'Code'
  if (language.toLowerCase() === 'js') return 'JavaScript'
  if (language.toLowerCase() === 'ts') return 'TypeScript'
  if (language.toLowerCase() === 'html') return 'HTML'
  if (language.toLowerCase() === 'css') return 'CSS'
  return language.toUpperCase()
}

const splitFenceInfo = (value) => {
  const trimmed = String(value || '').trim()
  if (!trimmed) return { language: 'Code', firstCodeLine: '' }
  const [, rawLanguage = '', firstCodeLine = ''] = trimmed.match(/^([a-zA-Z0-9_+#.-]+)?\s*([\s\S]*)$/) || []
  return {
    language: normalizeCodeLanguage(rawLanguage),
    firstCodeLine: firstCodeLine.trimStart(),
  }
}

// Shared by transcript rendering and the conversation file catalog.
export function codeFences(content) {
  const normalized = String(content || '').replace(/\r\n/g, '\n')
  const fences = []
  let cursor = 0
  let start = normalized.indexOf('```', cursor)
  while (start !== -1) {
    const infoStart = start + 3
    const nextLine = normalized.indexOf('\n', infoStart)
    const infoEnd = nextLine === -1 ? normalized.length : nextLine
    const { language, firstCodeLine } = splitFenceInfo(normalized.slice(infoStart, infoEnd))
    const codeStart = nextLine === -1 ? infoEnd : nextLine + 1
    const end = normalized.indexOf('```', codeStart)
    const body = normalized.slice(codeStart, end === -1 ? normalized.length : end)
    fences.push({ start, end: end === -1 ? normalized.length : end + 3, incomplete: end === -1, language,
      code: firstCodeLine ? `${firstCodeLine}${body ? `\n${body}` : ''}` : body })
    cursor = end === -1 ? normalized.length : end + 3
    start = normalized.indexOf('```', cursor)
  }
  return fences
}
