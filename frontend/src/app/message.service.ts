import {EventEmitter, inject, Injectable, Output} from '@angular/core';
import {HttpClient, HttpResponse} from "@angular/common/http";
import {Observable} from "rxjs";
import {MessageResponse} from "./models/message.response";
import {Urls} from "../utils/urls";
import {ChatResponse} from "../chat/models/chat.response";

@Injectable({
  providedIn: 'root'
})
export class MessageService {
    chat: ChatResponse | undefined = undefined;
    messages$: Array<MessageResponse> = [];
    @Output() newMessage = new EventEmitter();
    @Output() openedChat = new EventEmitter();    // used to update ChatService.openedChat without circular dependency

    constructor(private http: HttpClient) {
    }

    sendMessage(message: string): Observable<HttpResponse<string>> {
        let recipientName: string = this.chat!.otherMemberUsername;
        return this.http.post(Urls.sendMessage(recipientName), message, {observe: 'response', responseType: 'text'});
    }

    getMessagesByChat(chatId: number): Observable<Array<MessageResponse>> {
        return this.http.get<Array<MessageResponse>>(Urls.getMessagesInChat(chatId));
    }

    fetchMessages() {
        if (this.chat && this.chat.chatId)
            this.getMessagesByChat(this.chat.chatId).subscribe(messages => this.messages$ = messages);
        else
            this.messages$ = [];
    }

    public appendNewMessage(message: MessageResponse) {
        this.messages$.push(message);
        this.newMessage.emit();
    }

    showMessagesInChat(chat: ChatResponse) {
        this.chat = chat;
        this.openedChat.emit(chat.chatId? chat : undefined);
        this.fetchMessages();
    }

    hideMessagesInChat() {
        this.chat = undefined;
        this.openedChat.emit(undefined);
    }
}
