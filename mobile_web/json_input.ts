/** The shape of data before a protocol or storage boundary validates it. */
export type JsonInput = undefined | null | boolean | number | string | object;
export type JsonRecord = Record<string, JsonInput>;

export function parseJson(text: string): JsonInput {
  return JSON.parse(text) as JsonInput;
}

export function jsonRecord(value: JsonInput): JsonRecord | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord : null;
}
