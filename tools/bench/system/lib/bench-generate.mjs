import { ContractError, validateBenchGenerateRecord } from './contracts.mjs'

export class BenchGenerateParseError extends Error {
  constructor(code, message, options = {}) {
    super(message, options.cause ? { cause: options.cause } : undefined)
    this.name = 'BenchGenerateParseError'
    this.code = code
    this.line = options.line ?? null
  }
}

export function parseBenchGenerateJsonl(text, options = {}) {
  if (typeof text !== 'string') {
    throw new BenchGenerateParseError('invalid_parse', 'bench-generate stdout must be a string')
  }

  const records = []
  for (const [index, rawLine] of text.split(/\r?\n/).entries()) {
    if (rawLine.trim().length === 0) continue
    const line = index + 1
    let record
    try {
      record = JSON.parse(rawLine)
    } catch (error) {
      throw new BenchGenerateParseError(
        'invalid_parse',
        `bench-generate stdout line ${line} is not JSON: ${error.message}`,
        { cause: error, line },
      )
    }
    try {
      validateBenchGenerateRecord(record)
      validateThroughput(record)
    } catch (error) {
      if (error instanceof ContractError) {
        throw new BenchGenerateParseError(
          'invalid_contract',
          `bench-generate stdout line ${line} failed its contract: ${error.issues.join('; ')}`,
          { cause: error, line },
        )
      }
      if (error instanceof BenchGenerateParseError) {
        error.line ??= line
        throw error
      }
      throw error
    }
    validateIdentity(record, options, line)
    records.push(record)
  }

  if (records.length === 0) {
    throw new BenchGenerateParseError('empty_output', 'bench-generate stdout contained no records')
  }
  validateIterationSequence(records)
  return records
}

function validateIdentity(record, options, line) {
  if (options.expectedCommit !== undefined && record.commit !== options.expectedCommit) {
    throw new BenchGenerateParseError(
      'invalid_identity',
      `bench-generate stdout line ${line} commit ${JSON.stringify(record.commit)} does not match ${JSON.stringify(options.expectedCommit)}`,
      { line },
    )
  }
  if (options.expectedModel !== undefined && record.model !== options.expectedModel) {
    throw new BenchGenerateParseError(
      'invalid_identity',
      `bench-generate stdout line ${line} model ${JSON.stringify(record.model)} does not match ${JSON.stringify(options.expectedModel)}`,
      { line },
    )
  }
}

function validateIterationSequence(records) {
  for (let index = 0; index < records.length; index += 1) {
    if (records[index].iteration !== index) {
      throw new BenchGenerateParseError(
        'invalid_sequence',
        `bench-generate iteration ${records[index].iteration} appeared at record ${index}; expected ${index}`,
      )
    }
  }
}

function validateThroughput(record) {
  const decodeTokens = Math.max(0, record.generated_tokens - 1)
  const expected = record.decode_ms > 0 && decodeTokens > 0
    ? decodeTokens / (record.decode_ms / 1000)
    : 0
  if (!nearlyEqual(record.tokens_per_second, expected)) {
    throw new BenchGenerateParseError(
      'invalid_contract',
      `tokens_per_second ${record.tokens_per_second} does not match ${decodeTokens} decode tokens over ${record.decode_ms} ms (expected ${expected})`,
    )
  }
}

function nearlyEqual(actual, expected) {
  const scale = Math.max(1, Math.abs(actual), Math.abs(expected))
  return Math.abs(actual - expected) <= scale * 1e-9
}
