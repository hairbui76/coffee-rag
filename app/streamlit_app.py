"""Coffee Advisor — Streamlit chatbot UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Coffee Advisor",
    page_icon="☕",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    from src.pipeline import CoffeeRAG
    return CoffeeRAG()


def render_response(response) -> str:
    """Convert CoffeeResponse to markdown for display and history."""
    parts = [response.summary]

    if response.products:
        parts.append("\n**Sản phẩm gợi ý:**")
        for i, p in enumerate(response.products, 1):
            line = f"{i}. **{p.name}**"
            if p.roaster:
                line += f" — {p.roaster}"
            line += f"\n   {p.reason}"
            if p.url:
                line += f" [🔗 Xem]({p.url})"
            parts.append(line)

    if response.articles:
        parts.append("\n**Bài viết liên quan:**")
        for a in response.articles:
            parts.append(f"- **{a.title}**: {a.summary}")

    return "\n\n".join(parts)


def main():
    st.title("☕ Coffee Advisor")
    st.caption("Hệ thống tư vấn cà phê specialty — Hỏi bằng Tiếng Việt hoặc English")

    with st.sidebar:
        st.header("Về hệ thống")
        st.markdown(
            "**Coffee Advisor** sử dụng RAG (Retrieval-Augmented Generation) "
            "kết hợp dữ liệu 14,500+ sản phẩm cà phê specialty và 1,900+ bài báo "
            "để tư vấn cho bạn."
        )
        st.divider()
        st.markdown("**Ví dụ câu hỏi:**")
        examples = [
            "Gợi ý cà phê vị chocolate, rang vừa, từ Brazil",
            "Recommend a light roast Ethiopian coffee with fruity notes",
            "Natural process khác gì Washed?",
            "Tin tức mới nhất về thị trường cà phê Việt Nam",
            "So sánh Arabica với Robusta",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["pending_query"] = ex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_query", None)
    query = st.chat_input("Hỏi về cà phê... / Ask about coffee...") or pending

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        rag = load_pipeline()

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và phân tích..."):
                ctx = rag.retrieve(query)
                from src.generation.prompt_templates import build_prompt
                from src.generation.llm_client import generate_structured
                from src.generation.schemas import CoffeeResponse

                messages = build_prompt(query, ctx["beans"], ctx["news"])
                response = generate_structured(messages, CoffeeResponse, client=rag.llm_client)

            md = render_response(response)
            st.markdown(md)

        st.session_state.messages.append({"role": "assistant", "content": md})


if __name__ == "__main__":
    main()
