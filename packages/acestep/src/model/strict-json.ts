/** Parse one JSON value while rejecting duplicate object keys. */
export function parseStrictJson(text: string): unknown {
  return new StrictJsonParser(text).parse();
}

class StrictJsonParser {
  private offset = 0;

  constructor(private readonly text: string) {}

  parse(): unknown {
    this.skipWhitespace();
    const value = this.parseValue();
    this.skipWhitespace();
    if (this.offset !== this.text.length) this.fail("trailing content");
    return value;
  }

  private parseValue(): unknown {
    const character = this.text[this.offset];
    switch (character) {
      case "{":
        return this.parseObject();
      case "[":
        return this.parseArray();
      case '"':
        return this.parseString();
      case "t":
        return this.parseKeyword("true", true);
      case "f":
        return this.parseKeyword("false", false);
      case "n":
        return this.parseKeyword("null", null);
      default:
        if (character === "-" || isDigit(character)) return this.parseNumber();
        this.fail("expected a JSON value");
    }
  }

  private parseObject(): Record<string, unknown> {
    this.offset += 1;
    this.skipWhitespace();
    const result: Record<string, unknown> = Object.create(null) as Record<
      string,
      unknown
    >;
    const keys = new Set<string>();
    if (this.consume("}")) return result;
    while (true) {
      if (this.text[this.offset] !== '"') this.fail("expected an object key");
      const key = this.parseString();
      if (keys.has(key)) this.fail(`duplicate object key ${JSON.stringify(key)}`);
      keys.add(key);
      this.skipWhitespace();
      this.expect(":");
      this.skipWhitespace();
      result[key] = this.parseValue();
      this.skipWhitespace();
      if (this.consume("}")) return result;
      this.expect(",");
      this.skipWhitespace();
    }
  }

  private parseArray(): unknown[] {
    this.offset += 1;
    this.skipWhitespace();
    const result: unknown[] = [];
    if (this.consume("]")) return result;
    while (true) {
      result.push(this.parseValue());
      this.skipWhitespace();
      if (this.consume("]")) return result;
      this.expect(",");
      this.skipWhitespace();
    }
  }

  private parseString(): string {
    const start = this.offset;
    this.offset += 1;
    while (this.offset < this.text.length) {
      const character = this.text[this.offset]!;
      if (character === '"') {
        this.offset += 1;
        try {
          return JSON.parse(this.text.slice(start, this.offset)) as string;
        } catch {
          this.fail("invalid JSON string");
        }
      }
      if (character === "\\") {
        this.offset += 1;
        const escaped = this.text[this.offset];
        if (escaped === "u") {
          for (let index = 1; index <= 4; index += 1) {
            if (!isHex(this.text[this.offset + index])) {
              this.fail("invalid Unicode escape");
            }
          }
          this.offset += 5;
          continue;
        }
        if (!'"\\/bfnrt'.includes(escaped ?? "")) {
          this.fail("invalid string escape");
        }
        this.offset += 1;
        continue;
      }
      if (character.charCodeAt(0) <= 0x1f) {
        this.fail("unescaped control character in string");
      }
      this.offset += 1;
    }
    this.fail("unterminated string");
  }

  private parseNumber(): number {
    const remaining = this.text.slice(this.offset);
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(remaining);
    if (match === null) this.fail("invalid number");
    this.offset += match[0].length;
    const value = Number(match[0]);
    if (!Number.isFinite(value)) this.fail("non-finite number");
    return value;
  }

  private parseKeyword<T>(keyword: string, value: T): T {
    if (!this.text.startsWith(keyword, this.offset)) {
      this.fail(`expected ${keyword}`);
    }
    this.offset += keyword.length;
    return value;
  }

  private skipWhitespace(): void {
    while (
      this.text[this.offset] === " " ||
      this.text[this.offset] === "\n" ||
      this.text[this.offset] === "\r" ||
      this.text[this.offset] === "\t"
    ) {
      this.offset += 1;
    }
  }

  private consume(character: string): boolean {
    if (this.text[this.offset] !== character) return false;
    this.offset += 1;
    return true;
  }

  private expect(character: string): void {
    if (!this.consume(character)) this.fail(`expected ${JSON.stringify(character)}`);
  }

  private fail(reason: string): never {
    throw new SyntaxError(`Invalid JSON at offset ${this.offset}: ${reason}`);
  }
}

function isDigit(character: string | undefined): boolean {
  return character !== undefined && character >= "0" && character <= "9";
}

function isHex(character: string | undefined): boolean {
  return character !== undefined && /^[0-9a-f]$/i.test(character);
}
