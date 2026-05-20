from crewai import Agent, Task
from crewai.tools import tool
from duckduckgo_search import DDGS
import arxiv

# ----------------------------------------------------
# Custom Tools for Researcher Agent
# ----------------------------------------------------

@tool("DuckDuckGo Web Search")
def ddg_search(query: str) -> str:
    """Search the web using DuckDuckGo for general information, technical articles, and latest news related to the query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n-ve--")
            return "\n".join(formatted)
    except Exception as e:
        return f"Error during web search: {str(e)}"

@tool("ArXiv Academic Search")
def arxiv_search(query: str) -> str:
    """Search ArXiv database for scientific papers, academic preprints, abstracts, and reference details."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        formatted = []
        for r in search.results():
            authors = ", ".join([a.name for a in r.authors])
            formatted.append(
                f"Title: {r.title}\nAuthors: {authors}\nPublished: {r.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {r.summary}\nURL: {r.pdf_url}\n---"
            )
        if not formatted:
            return "No academic papers found for this query on ArXiv."
        return "\n".join(formatted)
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

# ----------------------------------------------------
# Agent Factories
# ----------------------------------------------------

def get_researcher(llm) -> Agent:
    return Agent(
        role="Lead Technical Researcher",
        goal="Perform in-depth literature search and compile factual evidence, abstracts, and key references for: {topic}",
        backstory=(
            "You are an elite research librarian and academic compiler. You excel at digging up primary research, "
            "synthesizing complex papers, and organizing technical facts. You verify facts and link findings to "
            "authoritative URLs and paper titles."
        ),
        tools=[ddg_search, arxiv_search],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def get_writer(llm) -> Agent:
    return Agent(
        role="Principal Scientific Writer",
        goal="Draft a comprehensive, rigorous, and highly informative technical research paper or article on: {topic}",
        backstory=(
            "You are a widely published academic author in computer science and engineering. You write in a "
            "formal, objective, and precise scientific style. You structure documents perfectly with clean Markdown "
            "and embed inline references mapped to the sources found by the research team."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def get_reviewer(llm) -> Agent:
    return Agent(
        role="Peer Review Committee Chair",
        goal="Rigorously evaluate the research paper for scientific accuracy, depth, organization, and adherence to original findings.",
        backstory=(
            "You are a strict editor-in-chief for a high-impact research journal. You despise hand-wavy claims, "
            "lack of detail, and formatting errors. You grade submissions strictly and provide direct, actionable "
            "feedback for revision."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def get_refiner(llm) -> Agent:
    return Agent(
        role="Senior Revision Editor",
        goal="Refine and revise the draft paper to resolve all items in the Peer Review feedback.",
        backstory=(
            "You are a master editor who specializes in refining scientific drafts. You know how to reorganize "
            "paragraphs for better flow, expand sections that are too thin, and address every piece of reviewer "
            "criticism while preserving academic style."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# ----------------------------------------------------
# Task Definitions
# ----------------------------------------------------

def get_research_task(agent, topic: str) -> Task:
    return Task(
        description=(
            f"Search for academic papers, articles, and authoritative documentation on the topic: '{topic}'.\n"
            "Gather at least 5 distinct high-quality sources.\n"
            "Summarize the key findings, methodologies, and relevant parameters from these sources.\n"
            "Produce a comprehensive, structured set of Research Notes.\n"
            "Include titles and URLs for all sources."
        ),
        expected_output="Detailed, structured research notes organized by sub-topics, including bibliography and source URLs.",
        agent=agent
    )

def get_drafting_task(agent, topic: str) -> Task:
    return Task(
        description=(
            f"Using the provided Research Notes, draft a publication-quality technical research paper on: '{topic}'.\n"
            "The paper MUST include these exact sections:\n"
            "1. Title\n"
            "2. Abstract (summary of findings and contributions)\n"
            "3. Introduction (context, problem statement, and importance)\n"
            "4. Literature Review / Current Landscape (citing sources from the notes)\n"
            "5. Methodology / Technical Architecture (explain how it works, technical specifics)\n"
            "6. Discussion & Practical Implications (detailed analysis)\n"
            "7. Conclusion & Future Directions\n"
            "8. References (with title, author, year, and URL if available)\n\n"
            "Formatting Rules:\n"
            "- Use clean Markdown (headings, lists, bold text, code blocks for technical details).\n"
            "- Write in a formal, objective, academic voice.\n"
            "- Do not write short summaries; expand each section to be detailed and informative.\n"
            "- Do not leave placeholder text or empty lists."
        ),
        expected_output="A complete, detailed technical research paper in Markdown format.",
        agent=agent
    )

def get_validation_task(agent, topic: str) -> Task:
    return Task(
        description=(
            "Review the generated technical research paper draft.\n"
            "Assess its quality, completeness, scientific accuracy, flow, and reference citations.\n\n"
            "You MUST output your evaluation in the following exact format (do not wrap in markdown or json blocks):\n\n"
            "SCORE: [Insert a single integer from 0 to 100]\n\n"
            "FEEDBACK:\n"
            "- [Bullet point of strengths]\n"
            "- [Bullet point of strengths]\n\n"
            "CRITICAL FIXES:\n"
            "- [Actionable instruction for what is missing or needs rewrite]\n"
            "- [Actionable instruction for what is missing or needs rewrite]"
        ),
        expected_output="An evaluation report formatted with SCORE, FEEDBACK, and CRITICAL FIXES sections.",
        agent=agent
    )

def get_refinement_task(agent, topic: str) -> Task:
    return Task(
        description=(
            "Revise the current research paper draft to address all items in the Peer Review feedback.\n"
            "Ensure all CRITICAL FIXES are implemented. Expand sections where requested, clarify arguments, "
            "and fix grammatical or formatting issues. Maintain the academic tone and Markdown format."
        ),
        expected_output="The fully revised and polished technical research paper in Markdown format.",
        agent=agent
    )
