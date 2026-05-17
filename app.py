import streamlit as st
from utils import extract_text_from_pdf, split_text_into_chunks
from rag_pipeline import (
    build_vector_store,
    retrieve_relevant_chunks,
    ask_groq_with_memory,
    generate_quiz,
    summarize_topic,
    summarize_full_pdf   
)

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")
st.caption("Upload your notes or textbook PDF and ask questions!")

# --- Session state init ---
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "revealed" not in st.session_state:
    st.session_state.revealed = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# --- PDF Upload (shared across all tabs) ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.index is None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing your PDF..."):
        raw_text = extract_text_from_pdf("temp.pdf")
        chunks = split_text_into_chunks(raw_text)
        index, chunks = build_vector_store(chunks)
        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success(f"PDF processed! {len(chunks)} chunks created.")

# --- Custom tab selector ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Chat with Notes", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 0 else "secondary"):
        st.session_state.active_tab = 0
        st.rerun()

with col2:
    if st.button("📝 Quiz Generator", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 1 else "secondary"):
        st.session_state.active_tab = 1
        st.rerun()

with col3:
    if st.button("📖 Topic Summarizer", use_container_width=True,
                 type="primary" if st.session_state.active_tab == 2 else "secondary"):
        st.session_state.active_tab = 2
        st.rerun()

st.divider()

# ── Tab 1: Chat with Memory ──────────────────────────────────
if st.session_state.active_tab == 0:
    if st.session_state.index is None:
        st.info("Upload a PDF above to get started.")
    else:
        st.subheader("Chat with Your Notes")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        query = st.chat_input("Ask anything about your notes...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_groq_with_memory(
                        query,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.chat_history
                    )
                st.write(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

# ── Tab 2: Quiz Generator ────────────────────────────────────
elif st.session_state.active_tab == 1:
    if st.session_state.index is None:
        st.info("Upload a PDF above to get started.")
    else:
        st.subheader("Generate a Quiz from Your Notes")

        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Enter a topic to quiz yourself on:",
                placeholder="e.g. machine learning, neural networks"
            )

        with col2:
            num_questions = st.selectbox(
                "Number of questions:",
                options=[3, 5, 10],
                index=1
            )

        if st.button("Generate Quiz", type="primary"):
            if not topic:
                st.warning("Please enter a topic first.")
            else:
                with st.spinner(f"Generating {num_questions} questions on '{topic}'..."):
                    questions = generate_quiz(
                        topic,
                        st.session_state.index,
                        st.session_state.chunks,
                        num_questions=num_questions
                    )

                if not questions:
                    st.error("Couldn't parse the quiz. Try a different topic or re-generate.")
                else:
                    st.session_state.quiz = questions
                    st.session_state.revealed = [False] * len(questions)

        if st.session_state.quiz:
            st.divider()
            st.markdown(f"### Quiz — {len(st.session_state.quiz)} Questions")

            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. {q['question']}**")

                for label, text in q["options"].items():
                    st.write(f"&nbsp;&nbsp;&nbsp;**{label}.** {text}")

                if st.button(f"Reveal Answer", key=f"reveal_{i}"):
                    st.session_state.revealed[i] = True

                if st.session_state.revealed[i]:
                    st.success(
                        f"Correct Answer: **{q['answer']}** — "
                        f"{q['options'][q['answer']]}"
                    )
                    st.caption(f"Explanation: {q['explanation']}")

                st.divider()

# ── Tab 3: Topic Summarizer ──────────────────────────────────
elif st.session_state.active_tab == 2:
    if st.session_state.index is None:
        st.info("Upload a PDF above to get started.")
    else:
        st.subheader("Summarize Your Notes")

        mode = st.radio(
            "What do you want to summarize?",
            options=["Specific Topic", "Entire PDF"],
            horizontal=True
        )

        st.divider()

        if mode == "Specific Topic":
            col1, col2 = st.columns(2)

            with col1:
                summary_topic = st.text_input(
                    "Enter a topic to summarize:",
                    placeholder="e.g. backpropagation, transformers"
                )

            with col2:
                summary_length = st.radio(
                    "Summary length:",
                    options=["short", "detailed"],
                    index=1,
                    horizontal=True
                )

            if st.button("Generate Summary", type="primary"):
                if not summary_topic:
                    st.warning("Please enter a topic first.")
                else:
                    with st.spinner(f"Summarizing '{summary_topic}'..."):
                        summary = summarize_topic(
                            summary_topic,
                            st.session_state.index,
                            st.session_state.chunks,
                            length=summary_length
                        )

                    st.divider()
                    st.markdown(f"### Summary: {summary_topic.title()}")
                    st.markdown(summary)

                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"summary_{summary_topic.replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

        else:
            st.info(
                f"This will sample chunks from across your entire PDF "
                f"({len(st.session_state.chunks)} chunks) and generate a full summary."
            )

            if st.button("Summarize Entire PDF", type="primary"):
                with st.spinner("Reading through your entire PDF... this may take a moment."):
                    full_summary = summarize_full_pdf(st.session_state.chunks)

                st.divider()
                st.markdown("### Full PDF Summary")
                st.markdown(full_summary)

                st.download_button(
                    label="Download Full Summary",
                    data=full_summary,
                    file_name="full_pdf_summary.txt",
                    mime="text/plain"
                )