import {MessageResponse} from "./message.response";
import {ChatResponse} from "../../chat/models/chat.response";

export class MessageWebsocketResponse {
    message!: MessageResponse;
    chatInContextOfSender!: ChatResponse;
    chatInContextOfRecipient!: ChatResponse;
}
