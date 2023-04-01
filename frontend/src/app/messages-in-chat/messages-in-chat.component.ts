import { Component } from '@angular/core';
import {faTrashAlt, faPaperPlane} from "@fortawesome/free-regular-svg-icons";
import {MessageService} from "../message.service";
import {ChatService} from "../../chat/chat.service";
import {ToastrService} from "ngx-toastr";
import {AuthService} from "../../auth/auth.service";
import {FormControl, FormGroup, Validators} from "@angular/forms";

@Component({
  selector: 'app-messages-in-chat',
  templateUrl: './messages-in-chat.component.html',
  styleUrls: ['./messages-in-chat.component.css']
})
export class MessagesInChatComponent {
    you: string = this.authService.getStoredCredentials()!.username;
    messageForm: FormGroup;
    sendIcon = faPaperPlane;
    deleteIcon = faTrashAlt;

    constructor(public messageService: MessageService, private chatService: ChatService,
                private authService: AuthService, private toastr: ToastrService) {
        messageService.newMessage.subscribe(() => {
            this.scrollToBottom();  // scroll to bottom when list of messages is updated
        });

        this.messageForm = new FormGroup({
            message: new FormControl('', Validators.required)
        });
    }

    sendMessage() {
        if (this.messageForm.get('message')?.valid) {
            this.messageService.sendMessage(this.messageForm.get('message')?.value).subscribe({
                next: () => {
                    this.messageForm.reset();
                    this.toastr.success("Sent", "", {timeOut: 500});
                },
                error: (error: { error: string | undefined; }) => {
                    this.toastr.error(error.error, 'Failed!', {timeOut: 5000});
                }
            })
        }
    }

    deleteChat() {
        if (this.messageService.chat && this.messageService.chat.chatId) {
            if (!confirm("Are you sure you want to delete this chat?"))
                return;

            this.chatService.deleteChat(this.messageService.chat!.chatId!).subscribe({
                next: () => {
                    this.toastr.success("Chat deleted!");
                },
                error: (error: { error: string | undefined; }) => {
                    this.toastr.error(error.error, 'Failed!', {timeOut: 5000});
                }
            });
        }
        else {
            this.messageService.hideMessagesInChat();
        }
    }

    scrollToBottom() {
        let messageList = document.getElementById("message-list");
        if (messageList)
            messageList.scrollTop = messageList.scrollHeight;
    }

    ngOnDestroy() {
        this.messageService.hideMessagesInChat();
    }
}
