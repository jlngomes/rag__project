"""
Interface Gradio — StrikeMetrics Solutions RAG CS:GO
Consome a API FastAPI em vez de chamar main.py diretamente.
"""

import os

import gradio as gr
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _get_models() -> list[str]:
    """Busca a lista de modelos disponíveis na API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["models"]
    except Exception as e:
        print(f"[app] Erro ao buscar modelos: {e}")
        return ["llama3.2"]  # fallback


def run_query(model_name: str, question: str) -> tuple[str, str]:
    """Envia a pergunta para a API e retorna (resposta, logs)."""
    if not question.strip():
        return "Por favor, insira uma pergunta antes de enviar.", ""

    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json={"model_name": model_name, "question": question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["answer"], data["logs"]
    except requests.exceptions.Timeout:
        return "Erro: a API demorou muito para responder (timeout).", ""
    except requests.exceptions.ConnectionError:
        return f"Erro: não foi possível conectar à API em {API_BASE_URL}.", ""
    except Exception as e:
        return f"Erro inesperado: {e}", ""


SUPPORTED_MODELS = _get_models()


def clear_all() -> tuple[str, str, str]:
    return "", "", ""


with gr.Blocks(title="StrikeMetrics Solutions — RAG CS:GO") as demo:

    gr.Markdown(
        """
        # StrikeMetrics Solutions — Análise de CS:GO com IA
        Faça perguntas em linguagem natural sobre estatísticas de combate.
        O sistema recupera contexto dos dados e gera uma resposta com o modelo selecionado.
        """
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300):
            model_dropdown = gr.Dropdown(
                choices=SUPPORTED_MODELS,
                value=SUPPORTED_MODELS[0],
                label="Modelo de IA (Ollama local)",
                info="Requer Ollama instalado. llama3.2 é o padrão recomendado.",
            )
            question_input = gr.Textbox(
                label="Pergunta",
                placeholder=(
                    "Ex: Qual arma causa mais dano médio na cabeça?\n"
                    "Ex: Como a AK-47 se compara à M4A1-S em dano por acerto?"
                ),
                lines=5,
            )
            with gr.Row():
                submit_btn = gr.Button("Enviar", variant="primary", scale=2)
                clear_btn  = gr.Button("Limpar", scale=1)

        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="Resposta da IA",
                lines=8,
                interactive=False,
            )
            logs_output = gr.Textbox(
                label="Logs do Pipeline RAG",
                lines=12,
                interactive=False,
                placeholder="Os logs de recuperação de documentos aparecerão aqui.",
            )

    submit_btn.click(
        fn=run_query,
        inputs=[model_dropdown, question_input],
        outputs=[answer_output, logs_output],
    )
    question_input.submit(
        fn=run_query,
        inputs=[model_dropdown, question_input],
        outputs=[answer_output, logs_output],
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[question_input, answer_output, logs_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
