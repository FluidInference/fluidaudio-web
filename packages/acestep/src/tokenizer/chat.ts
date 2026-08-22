export type AceChatRole = "system" | "user" | "assistant";

export interface AceChatMessage {
  readonly role: AceChatRole;
  readonly content: string;
  readonly reasoningContent?: string | null;
}

export interface AceChatTemplateOptions {
  readonly addGenerationPrompt?: boolean;
  readonly enableThinking?: boolean;
}

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";

/**
 * Exact no-tools subset of the pinned Qwen chat templates used by ACE-Step.
 *
 * ACE planner requests use system + user messages and an open assistant turn.
 * Assistant history is supported as well so captured planner continuations can
 * be replayed. Tool schemas/calls are intentionally rejected by the type and
 * are outside the Stage 1 product request contract.
 */
export function renderAceQwenChat(
  messages: readonly AceChatMessage[],
  options: AceChatTemplateOptions = {},
): string {
  if (messages.length === 0) throw new RangeError("Chat messages cannot be empty");
  for (const [index, message] of messages.entries()) {
    if (message.role !== "system" && message.role !== "user" && message.role !== "assistant") {
      throw new TypeError(`Unsupported chat role at index ${index}`);
    }
    if (typeof message.content !== "string") {
      throw new TypeError(`Chat message ${index} content must be a string`);
    }
  }

  let result = "";
  if (messages[0]!.role === "system") {
    result += `${IM_START}system\n${messages[0]!.content}${IM_END}\n`;
  }

  const lastQueryIndex = findLastQueryIndex(messages);
  messages.forEach((message, index) => {
    if (message.role === "system" && index === 0) return;
    if (message.role === "user" || message.role === "system") {
      result += `${IM_START}${message.role}\n${message.content}${IM_END}\n`;
      return;
    }

    let content = message.content;
    let reasoning = "";
    if (message.reasoningContent !== undefined && message.reasoningContent !== null) {
      reasoning = message.reasoningContent;
    } else if (content.includes("</think>")) {
      const beforeClosing = content.split("</think>", 1)[0]!;
      const openingParts = beforeClosing.split("<think>");
      reasoning = trimLeadingNewlines(trimTrailingNewlines(openingParts.at(-1)!));
      content = trimLeadingNewlines(content.split("</think>").at(-1)!);
    }

    result += `${IM_START}assistant\n`;
    if (index > lastQueryIndex && (index === messages.length - 1 || reasoning.length > 0)) {
      result += `<think>\n${trimNewlines(reasoning)}\n</think>\n\n${trimLeadingNewlines(content)}`;
    } else {
      result += content;
    }
    result += `${IM_END}\n`;
  });

  if (options.addGenerationPrompt === true) {
    result += `${IM_START}assistant\n`;
    if (options.enableThinking === false) result += "<think>\n\n</think>\n\n";
  }
  return result;
}

function findLastQueryIndex(messages: readonly AceChatMessage[]): number {
  let lastQueryIndex = messages.length - 1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!;
    if (
      message.role === "user" &&
      !(
        message.content.startsWith("<tool_response>") &&
        message.content.endsWith("</tool_response>")
      )
    ) {
      lastQueryIndex = index;
      break;
    }
  }
  return lastQueryIndex;
}

function trimNewlines(value: string): string {
  return trimLeadingNewlines(trimTrailingNewlines(value));
}

function trimLeadingNewlines(value: string): string {
  return value.replace(/^\n+/, "");
}

function trimTrailingNewlines(value: string): string {
  return value.replace(/\n+$/, "");
}
