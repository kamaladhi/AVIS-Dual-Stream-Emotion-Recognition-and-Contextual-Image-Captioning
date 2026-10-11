from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.utils.config import config
from loguru import logger
import mlflow

# ── Enable MLflow Tracing for GenAI ─────────────────────────
mlflow.langchain.autolog()

def generate_caption(state: AgentState) -> AgentState:
    logger.info("🤖 [Agent] Generating empathetic caption...")
    
    # Fallback to Hugging Face if explicitly requested via token
    if config.hf_token:
        # We use the HF Serverless API to prevent 16GB VRAM OOM crashes!
        from langchain_openai import ChatOpenAI
        logger.info("🤖 [Agent] Using Hugging Face Mistral-7B Serverless API...")
        llm = ChatOpenAI(
            api_key=config.hf_token,
            base_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1",
            model="tgi", # HF uses tgi for OpenAI schema
            temperature=0.4,
            max_tokens=150
        )
        
    elif config.openrouter_api_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=config.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3-8b-instruct", 
            temperature=0.4,
            max_tokens=150
        )
    elif config.groq_api_key:
        llm = ChatGroq(
            api_key=config.groq_api_key,
            model_name=config.llm_model_name,
            temperature=0.4,
            max_tokens=150
        )
    else:
        logger.warning("⚠️ [Agent] No API key found in .env! Returning raw concatenated string.")
        faces_str = ", ".join([f"{f['emotion']} (conf: {f['confidence']:.2f})" for f in state["faces_data"]])
        if not faces_str:
            faces_str = "No faces detected"
        state["final_caption"] = f"{state['base_caption']} [Emotions Detected: {faces_str}]"
        return state
        
    try:
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are AVIS, an empathetic and highly intelligent visual observer. Your goal is to synthesize raw visual data into a single, cohesive, human-sounding sentence. Do not act like a robot listing data points. Do not mention bounding boxes or confidence scores. Describe the scene naturally, smoothly blending the base context with the emotional state of the people involved."),
            ("user", "Base Scene Context: {base_caption}\n\nDetected People & Emotions: {faces_data}\n\nPlease generate a single, highly contextual, empathetic sentence describing this scene.")
        ])
        
        chain = prompt | llm
        
        # Clean up faces_data for the LLM (remove unnecessary bbox/confidence spam if desired, but we'll pass it for now)
        clean_faces = []
        for face in state["faces_data"]:
            clean_faces.append(f"Person expressing {face['emotion']}")
            
        faces_str = ", ".join(clean_faces) if clean_faces else "No people detected"
        
        response = chain.invoke({
            "base_caption": state["base_caption"],
            "faces_data": faces_str
        })
        
        state["final_caption"] = response.content.strip()
    except Exception as e:
        logger.error(f"❌ [Agent] LLM Generation failed: {e}")
        state["final_caption"] = state["base_caption"]
        
    return state

def build_caption_agent():
    """Builds and compiles the LangGraph State Machine."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate_caption", generate_caption)
    workflow.set_entry_point("generate_caption")
    workflow.add_edge("generate_caption", END)
    
    return workflow.compile()
    
# Compile a singleton instance to be imported by the pipeline
agent_app = build_caption_agent()
