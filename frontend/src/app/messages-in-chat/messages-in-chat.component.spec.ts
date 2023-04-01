import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MessagesInChatComponent } from './messages-in-chat.component';

describe('MessagesInChatComponent', () => {
  let component: MessagesInChatComponent;
  let fixture: ComponentFixture<MessagesInChatComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ MessagesInChatComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MessagesInChatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
