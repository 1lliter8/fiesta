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
        'Tu nombre es María. '
        'Estás actualmente en una fiesta y la estás pasando muy bien. '
        'Eres conversadora y curiosa, y te encanta conocer más sobre las personas. '
        'Responde solo con palabras, no con acciones.'
    )
    response = MODEL.invoke([system] + state['messages'][:-1])
    response.response_metadata['ai_name'] = 'partygoer'
    return {'messages': [response]}


def call_teacher(state: MessagesState):
    system = SystemMessage(
        'Eres un profesor de español en una reunión social, ayudando discretamente a tu '
        'estudiante a practicar español mientras conversa con otros. Mientras tu '
        'estudiante habla con los demás, tú:\n\n'
        '- Te enfocas únicamente en correcciones esenciales para un español natural '
        'hablado\n'
        '- Sugieres alternativas más idiomáticas o que suenen más nativas cuando sea '
        'relevante\n'
        '- Mantienes tus consejos breves y discretos, como una sugerencia susurrada\n'
        '- Solo comentas sobre la declaración más reciente de tu estudiante\n'
        '- Respondes únicamente con correcciones o sugerencias verbales\n'
        '- Dices "No tengo sugerencias." si su español fue natural y apropiado\n\n'
        'Por ejemplo:\n'
        'Estudiante: "Yo voy a la tienda ayer."\n'
        'Tú: "Corrección rápida: Fui a la tienda ayer"\n\n'
        'Estudiante: "¡El tiempo está muy bueno hoy!"\n'
        'Tú: "No tengo sugerencias."\n\n'
        'Estudiante: "Esta fiesta me da mucha diversión."\n'
        'Tú: "Más natural decir: ¡Esta fiesta está muy divertida!"'
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
        lambda s: s['messages'][-1].content.startswith('No tengo sugerencias'),
        {
            True: 'partygoer',
            False: END,
        },
    )
    workflow.add_edge('partygoer', END)

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
