from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph

load_dotenv()

MODEL = ChatAnthropic(
    model='claude-3-5-sonnet-20241022',
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


def call_partygoer(state: MessagesState):
    system = SystemMessage(
        'Your name is María.'
        "You're currently at a party and having a great time. "
        "You're chatty and inquisitive and love to find out more about people. "
        'Only response with speech, not actions.'
    )
    response = MODEL.invoke([system] + state['messages'][:-1])
    response.response_metadata['ai_name'] = 'partygoer'
    return {'messages': [response]}


def call_teacher(state: MessagesState):
    system = SystemMessage(
        'You are an English teacher at a social gathering, discreetly helping your '
        'student practice English while they converse with others. As their '
        'conversation partner speaks with others, you:\n\n'
        '- Focus solely on essential corrections for natural spoken English\n'
        '- Suggest more idiomatic or native-sounding alternatives when relevant\n'
        '- Keep your guidance brief and quiet, like a whispered suggestion\n'
        "- Only comment on your student's most recent statement\n"
        '- Respond only with verbal corrections or suggestions\n'
        "- Say 'I have no advice.' if their English was natural and appropriate\n\n"
        'For example:\n'
        "Student: 'I am going to the store yesterday.'\n"
        "You: 'Quick correction: I went to the store yesterday'\n\n"
        "Student: 'The weather is very good today!'\n"
        "You: 'I have no advice.'\n\n"
        "Student: 'This party gives me many fun.'\n"
        "You: 'More natural to say: This party is really fun!'"
    )
    response = MODEL.invoke([system] + state['messages'])
    response.response_metadata['ai_name'] = 'teacher'
    return {'messages': [response]}


def build_graph() -> CompiledGraph:
    workflow = StateGraph(MessagesState)

    workflow.add_node('teacher', call_teacher)
    workflow.add_node('partygoer', call_partygoer)

    workflow.add_edge(START, 'teacher')
    workflow.add_conditional_edges(
        'teacher',
        lambda s: s['messages'][-1].content.startswith('I have no advice'),
        {
            True: 'partygoer',
            False: END,
        },
    )
    workflow.add_edge('partygoer', END)

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
