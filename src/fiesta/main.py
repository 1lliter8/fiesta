import logging
from typing import Literal

from fasthtml import common as fh

from fiesta.graph import build_graph

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


# Headers and application

hdrs = hdrs = (
    fh.picolink,
    fh.Script(src='https://cdn.tailwindcss.com'),
    fh.Link(
        rel='stylesheet',
        href='https://cdn.jsdelivr.net/npm/daisyui@4.12.14/dist/full.min.css',
    ),
    fh.Script("""
        document.addEventListener('DOMContentLoaded', function() {
            function scrollChatToBottom() {
                requestAnimationFrame(() => {
                    const chatList = document.getElementById('chatlist');
                    if (chatList) {
                        chatList.scrollTop = chatList.scrollHeight;
                    }
                });
            }
            document.body.addEventListener('htmx:afterSwap', scrollChatToBottom);
            scrollChatToBottom();
        });
    """),
)
app = fh.FastHTML(hdrs=hdrs, cls='h-screen flex flex-col p-4 max-w-lg mx-auto')
fiesta = build_graph()


# Components


def chat_message(msg: str, user: Literal['user', 'partygoer', 'teacher']) -> fh.Div:
    """Renders a chat message button with the given message and user."""

    chat_class = 'chat-end'

    match user:
        case 'user':
            bubble_class = 'chat-bubble-primary'
            chat_header = 'You'
        case 'teacher':
            bubble_class = 'chat-bubble'
            chat_header = 'Teacher'
        case _:
            bubble_class = 'chat-bubble-secondary'
            chat_header = 'María'
            chat_class = 'chat-start'

    return fh.Div(cls=f'chat {chat_class}')(
        fh.Div(chat_header, cls='chat-header'),
        fh.Div(msg, cls=f'chat-bubble {bubble_class}'),
    )


def chat_input() -> fh.Input:
    """Renders the chat input field."""

    return fh.Input(
        name='msg',
        id='msg-input',
        placeholder='Type a message',
        cls='input input-bordered w-full',
        hx_swap_oob='true',
    )


# Render


@app.get
def index():
    chat_history = fh.Div(id='chatlist', cls='h-screen overflow-y-auto pb-24 px-4')
    input_area = fh.Form(
        fh.Div(cls='flex space-x-2')(
            fh.Group(chat_input(), fh.Button('Send', cls='btn btn-primary'))
        ),
        hx_post=send,
        hx_target='#chatlist',
        hx_swap='beforeend',
        cls='absolute bottom-0 left-0 right-0 p-4 bg-base-100',
    )
    page = fh.Div(cls='h-full relative')(chat_history, input_area)

    return fh.Titled('🥳 fiesta', page)


# Interaction


@app.post
def send(msg: str):
    r = fiesta.invoke(
        input={'messages': [msg]}, config={'configurable': {'thread_id': 42}}
    )

    r_msg = r['messages'][-1]

    logging.info(r_msg)

    return (
        chat_message(msg, 'user'),
        chat_message(r_msg.content, r_msg.response_metadata['ai_name']),
        chat_input(),
    )


# Serve


fh.serve()
