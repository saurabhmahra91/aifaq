"""
Professional Streamlit application for AI-powered FAQ Assistant.
"""

import logging

import plotly.express as px
import streamlit as st

# Import our modules
from aifaq import FAQRetriever, ResponseGenerator, ThresholdOptimizer, load_faq_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="AI FAQ Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .faq-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9fafb;
    }

    .confidence-high { color: #059669; font-weight: bold; }
    .confidence-medium { color: #d97706; font-weight: bold; }
    .confidence-low { color: #dc2626; font-weight: bold; }

    .sidebar-content {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_components():
    """
    Initialize FAQ retriever, generator, and optimizer components.
    """
    try:
        faq_data = load_faq_data("faqs.csv")
        retriever = FAQRetriever(faq_data)
        generator = ResponseGenerator()
        optimizer = ThresholdOptimizer()

        return retriever, generator, optimizer, faq_data
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()


def display_header():
    """
    Display the main application header.
    """
    st.markdown('<div class="main-header">AI FAQ Assistant</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div style="text-align: center; color: #6b7280; font-size: 1.2rem; margin-bottom: 2rem;">
        Intelligent FAQ retrieval with reinforcement learning optimization
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_metrics(retriever, optimizer, faq_data):
    """
    Display key metrics in the sidebar.
    """
    st.sidebar.markdown("## System Metrics")

    # FAQ Database Stats
    faq_stats = retriever.get_faq_statistics()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total FAQs", faq_stats["total_faqs"])
    with col2:
        st.metric("Cache Size", faq_stats["cache_size"])

    # RL Optimizer Stats
    rl_stats = optimizer.get_performance_stats()

    col3, col4 = st.sidebar.columns(2)
    with col3:
        st.metric("Best Threshold", f"{rl_stats['best_threshold']:.3f}")
    with col4:
        st.metric("Total Steps", rl_stats["total_steps"])

    # Current epsilon value
    st.sidebar.metric("Exploration Rate", f"{rl_stats['epsilon']:.3f}")

    # Reset RL button
    if st.sidebar.button("Reset RL", help="Reset reinforcement learning optimizer"):
        optimizer.reset()
        st.success("RL optimizer reset!")
        st.rerun()


def display_settings():
    """
    Display application settings in sidebar.
    """
    st.sidebar.markdown("## Settings")

    # Response mode selection
    response_mode = st.sidebar.selectbox(
        "Response Style", ["helpful", "concise", "detailed"], help="Choose how detailed you want the responses to be"
    )

    # Manual threshold override
    use_rl_threshold = st.sidebar.checkbox(
        "Use RL-optimized threshold", value=True, help="Use reinforcement learning to optimize the similarity threshold"
    )

    manual_threshold = None
    if not use_rl_threshold:
        manual_threshold = st.sidebar.slider(
            "Manual Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for FAQ matching",
        )

    # Number of FAQs to retrieve
    top_k = st.sidebar.slider(
        "Max FAQs to retrieve", min_value=1, max_value=5, value=3, help="Maximum number of similar FAQs to consider"
    )

    return response_mode, use_rl_threshold, manual_threshold, top_k


def display_query_interface(retriever, generator, optimizer, settings):
    """
    Display the main query interface.
    """
    response_mode, use_rl_threshold, manual_threshold, top_k = settings

    st.markdown("## Ask Your Question")

    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How can I reset my password??",
        help="Type any question and I'll find the most relevant FAQ information",
    )

    ask_button = st.button("🔍 Ask", type="primary")

    if query and ask_button:
        with st.spinner("🤔 Thinking..."):
            # Determine threshold
            threshold = optimizer.select_threshold() if use_rl_threshold else manual_threshold

            # Retrieve similar FAQs
            retrieved_faqs = retriever.retrieve_similar_faqs(query, threshold=threshold, top_k=top_k)

            # Generate response
            if retrieved_faqs:
                response_data = generator.generate_response_with_confidence(query, retrieved_faqs, response_mode)

                # Display results
                display_response_results(query, retrieved_faqs, response_data, threshold)

                # Collect user feedback for RL
                collect_user_feedback(query, threshold, optimizer)
            else:
                st.warning(
                    "No relevant FAQs found. Try rephrasing your question or lowering the similarity threshold."
                )


def display_response_results(query, retrieved_faqs, response_data, threshold):
    """
    Display the response results in an organized manner.
    """
    st.markdown("---")
    st.markdown("## Response")

    # Main response
    confidence = response_data["confidence"]
    confidence_class = get_confidence_class(confidence)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Generated Response:**")
        st.info(response_data["response"])

    with col2:
        st.markdown("**Confidence Score:**")
        st.markdown(f'<span class="{confidence_class}">{confidence:.2f}</span>', unsafe_allow_html=True)

        st.markdown(f"**Threshold Used:** {threshold:.3f}")
        st.markdown(f"**Sources:** {response_data['num_sources']}")

    # Retrieved FAQs
    if retrieved_faqs:
        st.markdown("### Retrieved FAQs")

        for i, item in enumerate(retrieved_faqs, 1):
            faq = item["faq"]
            similarity = item["similarity"]

            with st.expander(f"FAQ {i} - Similarity: {similarity:.3f}"):
                st.markdown(f"**Q:** {faq['question']}")
                st.markdown(f"**A:** {faq['answer']}")


def collect_user_feedback(query, threshold, optimizer):
    """
    Collect user feedback for reinforcement learning.
    """
    st.markdown("### Was this response helpful? (Your response will be used to improve the system)")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("👍 Yes", key=f"positive_{query}"):
            optimizer.update_reward(threshold, 1.0)
            print(optimizer.q_values)
            st.success("Thank you for your feedback!")
            st.rerun()

    with col2:
        if st.button("👎 No", key=f"negative_{query}"):
            optimizer.update_reward(threshold, 0.0)
            print(optimizer.q_values)
            st.success("Thank you for your feedback! I'll improve.")
            st.rerun()


def get_confidence_class(confidence):
    """
    Get CSS class based on confidence level.
    """
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"


def _populate_demo_data(optimizer):
    """
    Populate optimizer with some demo training data.
    """

    # Simulate 15 interactions with varying rewards
    demo_interactions = [
        (0.7, 1.0), (0.6, 0.0), (0.8, 1.0), (0.5, 0.0), (0.75, 1.0),
        (0.7, 1.0), (0.65, 1.0), (0.4, 0.0), (0.8, 1.0), (0.7, 1.0),
        (0.6, 0.0), (0.75, 1.0), (0.85, 1.0), (0.5, 0.0), (0.7, 1.0)
    ]

    for threshold, reward in demo_interactions:
        # Find closest available threshold
        distances = [abs(t - threshold) for t in optimizer.thresholds]
        closest_idx = distances.index(min(distances))
        closest_threshold = optimizer.thresholds[closest_idx]

        optimizer.update_reward(closest_threshold, reward)

    logger.info("Populated optimizer with demo data")


def display_analytics_tab(optimizer, retriever):
    """
    Display analytics and performance metrics.
    """
    st.markdown("## Analytics Dashboard")

    # RL Performance
    rl_stats = optimizer.get_performance_stats()
    has_training_data = rl_stats["total_steps"] > 0
    has_feedback_data = any(q > 0 for q in rl_stats["q_values"])
    has_usage_data = any(count > 0 for count in rl_stats["action_counts"])

    if not has_training_data:
        st.info(
            "**No activity yet!** Start asking questions to see threshold usage patterns."
        )
    elif not has_feedback_data:
        st.info(
            "**Questions asked but no feedback yet!** Provide feedback (👍/👎) on responses "
            "to see the reinforcement learning optimization in action."
        )

    # Add button to simulate some demo data if no meaningful data exists
    if not has_feedback_data and st.button("🎮 Generate Demo Data", help="Add some sample training data to see how analytics look"):
        _populate_demo_data(optimizer)
        st.success("Demo data generated! Refresh to see the analytics in action.")
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        # Q-values visualization
        if has_feedback_data:
            fig_q = px.bar(
                x=rl_stats["thresholds"],
                y=rl_stats["q_values"],
                title="Q-values by Threshold",
                labels={"x": "Threshold", "y": "Q-value"},
                color=rl_stats["q_values"],
                color_continuous_scale="viridis",
            )
            fig_q.update_layout(showlegend=False, coloraxis_showscale=False)
        else:
            # Show current state or demo
            demo_values = [0.5] * len(rl_stats["thresholds"]) if not has_training_data else rl_stats["q_values"]
            title_suffix = " (Demo)" if not has_training_data else " (No Feedback Yet)"

            fig_q = px.bar(
                x=rl_stats["thresholds"],
                y=demo_values,
                title=f"Q-values by Threshold{title_suffix}",
                labels={"x": "Threshold", "y": "Q-value"},
                opacity=0.3 if not has_training_data else 0.7,
            )
            fig_q.update_layout(showlegend=False)

            annotation_text = ("Ask questions and provide feedback<br>to see real data here!"
                              if not has_training_data
                              else "Provide feedback (👍/👎)<br>to see learning progress!")
            fig_q.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )

        st.plotly_chart(fig_q, use_container_width=True)

    with col2:
        # Action counts
        if has_usage_data:
            fig_counts = px.bar(
                x=rl_stats["thresholds"],
                y=rl_stats["action_counts"],
                title="Threshold Usage Frequency",
                labels={"x": "Threshold", "y": "Usage Count"},
                color=rl_stats["action_counts"],
                color_continuous_scale="blues",
            )
            fig_counts.update_layout(showlegend=False, coloraxis_showscale=False)
        else:
            # Show demo data
            demo_counts = [2] * len(rl_stats["thresholds"]) if not has_training_data else rl_stats["action_counts"]
            title_suffix = " (Demo)" if not has_training_data else ""

            fig_counts = px.bar(
                x=rl_stats["thresholds"],
                y=demo_counts,
                title=f"Threshold Usage Frequency{title_suffix}",
                labels={"x": "Threshold", "y": "Usage Count"},
                opacity=0.3 if not has_training_data else 1.0,
            )
            fig_counts.update_layout(showlegend=False)

            if not has_training_data:
                fig_counts.add_annotation(
                    text="Usage patterns will appear<br>as you interact with the system",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )

        st.plotly_chart(fig_counts, use_container_width=True)

    # Learning Progress Section
    st.markdown("### Learning Progress")

    if has_usage_data:
        progress_col1, progress_col2, progress_col3 = st.columns(3)

        with progress_col1:
            st.metric(
                "Training Steps",
                rl_stats["total_steps"],
                help="Number of user interactions used for learning"
            )

        with progress_col2:
            exploration_rate = rl_stats["epsilon"] * 100
            st.metric(
                "Exploration Rate",
                f"{exploration_rate:.1f}%",
                help="Percentage of time the system explores new thresholds vs exploits known good ones"
            )

        with progress_col3:
            best_threshold = rl_stats["best_threshold"]
            st.metric(
                "Best Threshold",
                f"{best_threshold:.3f}",
                help="Currently best performing similarity threshold"
            )

        # Learning insights
        if rl_stats["total_steps"] >= 5:
            total_interactions = sum(rl_stats["action_counts"])
            if total_interactions > 0:
                most_used_idx = rl_stats["action_counts"].index(max(rl_stats["action_counts"]))
                most_used_threshold = rl_stats["thresholds"][most_used_idx]

                best_q_idx = rl_stats["q_values"].index(max(rl_stats["q_values"]))
                best_q_threshold = rl_stats["thresholds"][best_q_idx]

                if most_used_threshold == best_q_threshold:
                    st.success("🎯 **System is converging!** Most used threshold matches best performing threshold.")
                else:
                    st.info(
                        f"🔍 **Still learning:** Most used ({most_used_threshold:.3f}) "
                        f"vs best ({best_q_threshold:.3f}) threshold."
                    )
        else:
            st.info("💡 **Tip:** Provide more feedback (👍/👎) to help the system learn your preferences!")
    else:
        st.markdown("""
        **How the learning works:**
        1. System tries different similarity thresholds
        2. You provide feedback on response quality
        3. AI learns which thresholds work best
        4. System gets better at finding relevant FAQs
        """)

    # System Statistics
    st.markdown("### System Statistics")

    faq_stats = retriever.get_faq_statistics()

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("Total FAQs", faq_stats["total_faqs"])

    with metrics_col2:
        st.metric("Cache Size", faq_stats["cache_size"])

    with metrics_col3:
        st.metric("Avg Q Length", f"{faq_stats['avg_question_length']:.0f}")

    with metrics_col4:
        st.metric("Avg A Length", f"{faq_stats['avg_answer_length']:.0f}")


def display_faq_management_tab(faq_data):
    """
    Display FAQ management interface.
    """
    st.markdown("## FAQ Database")

    # Add search functionality
    search_term = st.text_input("Search FAQs", placeholder="Enter keywords to search...")

    # Filter FAQs based on search term
    if search_term:
        search_lower = search_term.lower()
        filtered_faqs = [
            faq for faq in faq_data
            if search_lower in faq["question"].lower() or search_lower in faq["answer"].lower()
        ]
    else:
        filtered_faqs = faq_data

    # Display FAQs
    st.markdown(f"### Showing {len(filtered_faqs)} of {len(faq_data)} FAQs")

    for idx, faq in enumerate(filtered_faqs):
        with st.expander(f"FAQ {idx + 1}: {faq['question'][:60]}..."):
            st.markdown(f"**Question:** {faq['question']}")
            st.markdown(f"**Answer:** {faq['answer']}")


def main():
    """
    Main application function.
    """
    # Initialize components
    retriever, generator, optimizer, faq_data = initialize_components()

    # Display header
    display_header()

    # Sidebar
    with st.sidebar:
        display_metrics(retriever, optimizer, faq_data)
        settings = display_settings()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "📋 FAQ Database"])

    with tab1:
        display_query_interface(retriever, generator, optimizer, settings)

    with tab2:
        display_analytics_tab(optimizer, retriever)

    with tab3:
        display_faq_management_tab(faq_data)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        AI FAQ Assistant with Reinforcement Learning | Built with Streamlit & LiteLLM
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
