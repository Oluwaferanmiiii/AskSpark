"""
Deep Research Agent System for AskSpark
Week 2 Lab 4 - Advanced research automation with multi-source intelligence
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from agents import Agent, Runner, trace, function_tool, WebSearchTool, gen_trace_id
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field, validator
import os

from ..base_agent import AskSparkAgentBase, AgentResponse
from ..tools import AskSparkTools
from ..structured_outputs import (
    StructuredOutputsManager,
    DocumentInsightStructured,
    ContentSafetyLevel
)
from ...core.ai_providers import UnifiedAIClient
from ...config.settings import Config
from ...config.logging import get_logger

logger = get_logger(__name__)

# Research Data Models
class ResearchSource(Enum):
    """Research source types"""
    WEB_SEARCH = "web_search"
    DOCUMENT_ANALYSIS = "document_analysis"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    EXPERT_CONSULTATION = "expert_consultation"

class ResearchDepth(Enum):
    """Research depth levels"""
    SURFACE = "surface"
    MODERATE = "moderate"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"

@dataclass
class ResearchQuery:
    """Research query structure"""
    query_id: str
    topic: str
    research_question: str
    depth: ResearchDepth
    sources: List[ResearchSource]
    constraints: List[str]
    expected_output_format: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResearchFinding:
    """Individual research finding"""
    finding_id: str
    query_id: str
    source: ResearchSource
    content: str
    confidence_score: float
    relevance_score: float
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    publication_date: Optional[datetime] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResearchReport:
    """Complete research report"""
    report_id: str
    query_id: str
    topic: str
    executive_summary: str
    key_findings: List[ResearchFinding]
    detailed_analysis: str
    insights_and_recommendations: List[str]
    data_visualizations: List[Dict[str, Any]]
    limitations: List[str]
    confidence_level: str
    completion_percentage: float
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class DeepResearchAgent(AskSparkAgentBase):
    """Agent for conducting deep research across multiple sources"""
    
    def __init__(self):
        instructions = """
        You are a deep research specialist with expertise in comprehensive information gathering and analysis.
        Your task is to:
        1. Conduct thorough research across multiple sources
        2. Synthesize information from diverse sources
        3. Identify patterns, trends, and insights
        4. Provide actionable recommendations
        5. Maintain high standards of accuracy and reliability
        
        Research Methodology:
        - Use multiple, credible sources
        - Cross-reference and validate information
        - Identify conflicting or biased information
        - Extract specific data points and statistics
        - Provide context and background information
        
        Quality Standards:
        - Prioritize recent and relevant information
        - Distinguish between facts and opinions
        - Provide source attribution and credibility assessment
        - Identify knowledge gaps and areas of uncertainty
        - Offer balanced, objective analysis
        
        Research Areas:
        - Market research and industry analysis
        - Competitive intelligence
        - Technology trends and innovations
        - Regulatory and compliance information
        - Financial and business data
        - Academic and expert insights
        
        Always provide structured, well-documented research findings.
        """
        super().__init__("DeepResearchAgent", instructions)
        self.web_search_tool = WebSearchTool()
    
    async def research_topic(self, query: ResearchQuery) -> AgentResponse:
        """Conduct deep research on a specific topic"""
        input_text = f"""
        Conduct comprehensive deep research on:
        
        Topic: {query.topic}
        Research Question: {query.research_question}
        Depth Level: {query.depth.value}
        Sources: {[source.value for source in query.sources]}
        Constraints: {', '.join(query.constraints)}
        
        Expected Output: {query.expected_output_format}
        
        Please provide:
        1. Executive summary of key findings
        2. Detailed analysis with supporting data
        3. Insights and actionable recommendations
        4. Source attribution and credibility assessment
        5. Limitations and areas for further research
        
        Focus on accuracy, relevance, and comprehensive coverage.
        """
        return await self.run(input_text)
    
    async def synthesize_findings(self, findings: List[ResearchFinding], 
                                  synthesis_question: str) -> AgentResponse:
        """Synthesize multiple research findings"""
        findings_text = "\n\n".join([
            f"Finding {i+1}: {finding.content} (Source: {finding.source.value}, Confidence: {finding.confidence_score})"
            for i, finding in enumerate(findings)
        ])
        
        input_text = f"""
        Synthesize the following research findings to answer: {synthesis_question}
        
        Research Findings:
        {findings_text}
        
        Please provide:
        1. Integrated analysis addressing the synthesis question
        2. Key themes and patterns identified
        3. Conflicting information and resolutions
        4. Confidence assessment for synthesized conclusions
        5. Recommendations based on integrated findings
        
        Focus on creating a coherent, comprehensive synthesis.
        """
        return await self.run(input_text)
    
    async def generate_research_plan(self, topic: str, research_objectives: List[str]) -> AgentResponse:
        """Generate a comprehensive research plan"""
        objectives_text = "\n".join([f"- {obj}" for obj in research_objectives])
        
        input_text = f"""
        Generate a comprehensive research plan for:
        
        Topic: {topic}
        Research Objectives:
        {objectives_text}
        
        Please provide a detailed research plan including:
        1. Research methodology and approach
        2. Primary and secondary sources to explore
        3. Search strategy and keywords
        4. Data collection and validation methods
        5. Analysis framework and criteria
        6. Timeline and resource requirements
        7. Risk mitigation and quality control measures
        
        Focus on thoroughness, efficiency, and reliability.
        """
        return await self.run(input_text)

class ClarificationAgent(AskSparkAgentBase):
    """Agent for asking clarifying questions to improve research"""
    
    def __init__(self):
        instructions = """
        You are a research clarification specialist. Your task is to:
        1. Identify ambiguities and gaps in research requests
        2. Ask targeted clarifying questions
        3. Suggest refinements to research scope
        4. Ensure research objectives are clear and achievable
        5. Help define success criteria
        
        Clarification Focus Areas:
        - Scope and boundaries of research
        - Specific aspects or subtopics to prioritize
        - Required level of detail and depth
        - Target audience and use case for research
        - Time constraints and urgency
        - Available resources and access limitations
        
        Question Guidelines:
        - Ask specific, targeted questions
        - Provide options when appropriate
        - Explain why clarification is needed
        - Suggest practical refinements
        - Maintain focus on research objectives
        
        Always aim to improve research effectiveness and efficiency.
        """
        super().__init__("ClarificationAgent", instructions)
    
    async def analyze_research_request(self, topic: str, objectives: List[str]) -> AgentResponse:
        """Analyze research request and ask clarifying questions"""
        objectives_text = "\n".join([f"- {obj}" for obj in objectives])
        
        input_text = f"""
        Analyze this research request and provide clarifying questions:
        
        Topic: {topic}
        Objectives:
        {objectives_text}
        
        Please identify:
        1. Ambiguities or unclear aspects
        2. Potential scope issues or overlaps
        3. Missing information or constraints
        4. Specific clarifying questions needed
        5. Suggestions for improving the research request
        
        Focus on making the research more precise and effective.
        """
        return await self.run(input_text)
    
    async def refine_research_query(self, original_query: str, 
                                  clarifications: Dict[str, str]) -> AgentResponse:
        """Refine research query based on clarifications"""
        clarifications_text = "\n".join([f"- {k}: {v}" for k, v in clarifications.items()])
        
        input_text = f"""
        Refine the research query based on clarifications:
        
        Original Query: {original_query}
        Clarifications:
        {clarifications_text}
        
        Provide a refined research query that:
        1. Incorporates all clarifications provided
        2. Is more specific and focused
        3. Maintains the original intent
        4. Defines clear scope and boundaries
        5. Specifies expected outcomes
        
        Explain the improvements made and why they matter.
        """
        return await self.run(input_text)

class ResearchQualityAgent(AskSparkAgentBase):
    """Agent for evaluating and ensuring research quality"""
    
    def __init__(self):
        instructions = """
        You are a research quality evaluation specialist. Your task is to:
        1. Assess the quality and reliability of research findings
        2. Identify potential biases or limitations
        3. Evaluate source credibility and relevance
        4. Check for logical consistency and completeness
        5. Provide quality improvement recommendations
        
        Quality Assessment Criteria:
        - Source credibility and authority
        - Information recency and relevance
        - Methodology soundness
        - Logical consistency and coherence
        - Completeness and depth
        - Objectivity and bias identification
        - Actionability of recommendations
        
        Quality Levels:
        - Excellent: High credibility, comprehensive, actionable
        - Good: Reliable sources, mostly complete, useful
        - Fair: Some limitations, partially complete
        - Poor: Major issues, incomplete, unreliable
        
        Always provide specific, constructive feedback for improvement.
        """
        super().__init__("ResearchQualityAgent", instructions)
    
    async def evaluate_research_quality(self, research_report: ResearchReport) -> AgentResponse:
        """Evaluate the quality of a research report"""
        findings_text = "\n".join([f"- {finding.content}" for finding in research_report.key_findings[:5]])
        
        input_text = f"""
        Evaluate the quality of this research report:
        
        Topic: {research_report.topic}
        Executive Summary: {research_report.executive_summary[:200]}...
        Key Findings: {findings_text}
        Confidence Level: {research_report.confidence_level}
        Completion: {research_report.completion_percentage}%
        
        Please assess:
        1. Overall quality score (1-10)
        2. Source credibility and diversity
        3. Information recency and relevance
        4. Logical consistency and coherence
        5. Completeness and depth
        6. Objectivity and bias assessment
        7. Actionability of recommendations
        8. Specific improvement recommendations
        
        Provide detailed evaluation with specific examples.
        """
        return await self.run(input_text)
    
    async def suggest_improvements(self, quality_evaluation: str, 
                                  research_report: ResearchReport) -> AgentResponse:
        """Suggest specific improvements for research"""
        input_text = f"""
        Based on this quality evaluation, suggest specific improvements:
        
        Quality Evaluation: {quality_evaluation}
        Research Topic: {research_report.topic}
        Current Confidence: {research_report.confidence_level}
        Current Completion: {research_report.completion_percentage}%
        
        Please provide:
        1. Priority improvement areas
        2. Specific action items for each area
        3. Additional sources or methods to consider
        4. Quality control measures to implement
        5. Timeline for implementing improvements
        
        Focus on practical, achievable improvements.
        """
        return await self.run(input_text)

class DeepResearchTools:
    """Tools for deep research operations"""
    
    def __init__(self):
        self.deep_research_agent = DeepResearchAgent()
        self.clarification_agent = ClarificationAgent()
        self.quality_agent = ResearchQualityAgent()
        self.structured_manager = StructuredOutputsManager()
        self.web_search_tool = WebSearchTool()
        self.tools = AskSparkTools()
        self.research_database = {}  # In-memory for demo
        self.reports_database = {}
        logger.info("Initialized Deep Research Tools")
    
    @function_tool
    async def create_research_query(self, topic: str, research_question: str,
                                 depth: str, sources: List[str],
                                 constraints: List[str], output_format: str) -> Dict[str, Any]:
        """Create a new research query"""
        try:
            # Validate inputs
            if depth not in ["surface", "moderate", "deep", "comprehensive"]:
                depth = "moderate"  # Default
            
            query = ResearchQuery(
                query_id=f"query_{len(self.research_database) + 1}",
                topic=topic,
                research_question=research_question,
                depth=ResearchDepth(depth),
                sources=[ResearchSource(source) for source in sources],
                constraints=constraints,
                expected_output_format=output_format
            )
            
            # Store in database
            self.research_database[query.query_id] = query
            
            return {
                "success": True,
                "query_id": query.query_id,
                "query": {
                    "topic": query.topic,
                    "research_question": query.research_question,
                    "depth": query.depth.value,
                    "sources": [source.value for source in query.sources],
                    "constraints": query.constraints,
                    "expected_output_format": query.expected_output_format,
                    "created_at": query.created_at.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Failed to create research query: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def conduct_web_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Conduct web search for research"""
        try:
            # Use web search tool
            search_results = await self.web_search_tool.search(query, max_results=max_results)
            
            # Process results
            findings = []
            for i, result in enumerate(search_results[:max_results]):
                finding = ResearchFinding(
                    finding_id=f"web_{i+1}",
                    query_id="web_search",
                    source=ResearchSource.WEB_SEARCH,
                    content=result.get("content", ""),
                    confidence_score=0.8,  # Default confidence
                    relevance_score=1.0 - (i * 0.1),  # Decreasing relevance
                    source_url=result.get("url"),
                    source_title=result.get("title"),
                    extracted_data=result
                )
                findings.append(finding)
            
            return {
                "success": True,
                "query": query,
                "results_count": len(findings),
                "findings": [
                    {
                        "finding_id": f.finding_id,
                        "content": f.content,
                        "source": f.source.value,
                        "confidence": f.confidence_score,
                        "relevance": f.relevance_score,
                        "url": f.source_url,
                        "title": f.source_title
                    }
                    for f in findings
                ]
            }
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def analyze_documents_for_research(self, documents: List[str], 
                                       research_question: str) -> Dict[str, Any]:
        """Analyze documents for research insights"""
        try:
            findings = []
            
            for i, doc in enumerate(documents):
                # Use structured document analysis
                analysis_result = await self.structured_manager.process_lead_with_structured_output(
                    doc, research_question
                )
                
                if analysis_result.get("success"):
                    structured_data = analysis_result["structured_data"]
                    
                    finding = ResearchFinding(
                        finding_id=f"doc_{i+1}",
                        query_id="document_analysis",
                        source=ResearchSource.DOCUMENT_ANALYSIS,
                        content=structured_data.get("summary", ""),
                        confidence_score=structured_data.get("confidence_score", 0.7),
                        relevance_score=0.9,  # High relevance for direct analysis
                        extracted_data=structured_data
                    )
                    findings.append(finding)
            
            return {
                "success": True,
                "research_question": research_question,
                "documents_analyzed": len(documents),
                "findings": [
                    {
                        "finding_id": f.finding_id,
                        "content": f.content,
                        "source": f.source.value,
                        "confidence": f.confidence_score,
                        "relevance": f.relevance_score,
                        "data": f.extracted_data
                    }
                    for f in findings
                ]
            }
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def synthesize_research_findings(self, findings: List[Dict[str, Any]], 
                                       synthesis_question: str) -> Dict[str, Any]:
        """Synthesize multiple research findings"""
        try:
            # Convert to ResearchFinding objects
            research_findings = []
            for i, finding_data in enumerate(findings):
                finding = ResearchFinding(
                    finding_id=finding_data.get("finding_id", f"synth_{i+1}"),
                    query_id="synthesis",
                    source=ResearchSource(finding_data.get("source", "web_search")),
                    content=finding_data.get("content", ""),
                    confidence_score=finding_data.get("confidence", 0.7),
                    relevance_score=finding_data.get("relevance", 0.8),
                    extracted_data=finding_data.get("data", {})
                )
                research_findings.append(finding)
            
            # Use deep research agent for synthesis
            synthesis_result = await self.deep_research_agent.synthesize_findings(
                research_findings, synthesis_question
            )
            
            return {
                "success": True,
                "synthesis_question": synthesis_question,
                "synthesized_content": synthesis_result.content,
                "execution_time": synthesis_result.execution_time,
                "model_used": synthesis_result.model_used
            }
        except Exception as e:
            logger.error(f"Research synthesis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def create_research_report(self, query_id: str, findings: List[Dict[str, Any]],
                                   synthesis: str, analysis: str) -> Dict[str, Any]:
        """Create a comprehensive research report"""
        try:
            # Convert to ResearchFinding objects
            research_findings = []
            for finding_data in findings:
                finding = ResearchFinding(
                    finding_id=finding_data.get("finding_id", ""),
                    query_id=query_id,
                    source=ResearchSource(finding_data.get("source", "web_search")),
                    content=finding_data.get("content", ""),
                    confidence_score=finding_data.get("confidence", 0.7),
                    relevance_score=finding_data.get("relevance", 0.8),
                    extracted_data=finding_data.get("data", {})
                )
                research_findings.append(finding)
            
            # Get original query
            original_query = self.research_database.get(query_id)
            topic = original_query.topic if original_query else "Unknown Topic"
            
            # Create report
            report = ResearchReport(
                report_id=f"report_{len(self.reports_database) + 1}",
                query_id=query_id,
                topic=topic,
                executive_summary=synthesis[:500],  # Truncate for summary
                key_findings=research_findings,
                detailed_analysis=analysis,
                insights_and_recommendations=[],  # Would be extracted from analysis
                data_visualizations=[],
                limitations=["Based on available sources", "Time constraints"],
                confidence_level="High" if all(f.confidence_score > 0.7 for f in research_findings) else "Medium",
                completion_percentage=100.0
            )
            
            # Store report
            self.reports_database[report.report_id] = report
            
            return {
                "success": True,
                "report_id": report.report_id,
                "report": {
                    "report_id": report.report_id,
                    "query_id": query_id,
                    "topic": report.topic,
                    "executive_summary": report.executive_summary,
                    "key_findings_count": len(report.key_findings),
                    "confidence_level": report.confidence_level,
                    "completion_percentage": report.completion_percentage,
                    "created_at": report.created_at.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Report creation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def get_research_report(self, report_id: str) -> Dict[str, Any]:
        """Retrieve a research report"""
        try:
            report = self.reports_database.get(report_id)
            if not report:
                return {"success": False, "error": "Report not found"}
            
            return {
                "success": True,
                "report": {
                    "report_id": report.report_id,
                    "query_id": report.query_id,
                    "topic": report.topic,
                    "executive_summary": report.executive_summary,
                    "detailed_analysis": report.detailed_analysis,
                    "insights_count": len(report.insights_and_recommendations),
                    "limitations": report.limitations,
                    "confidence_level": report.confidence_level,
                    "completion_percentage": report.completion_percentage,
                    "created_at": report.created_at.isoformat(),
                    "last_updated": report.last_updated.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Report retrieval failed: {str(e)}")
            return {"success": False, "error": str(e)}

class DeepResearchManager:
    """Manager for coordinating deep research operations"""
    
    def __init__(self):
        self.deep_research_agent = DeepResearchAgent()
        self.clarification_agent = ClarificationAgent()
        self.quality_agent = ResearchQualityAgent()
        self.tools = DeepResearchTools()
        self.unified_client = UnifiedAIClient()
        
        logger.info("Initialized Deep Research Manager")
    
    async def process_research_request(self, topic: str, objectives: List[str],
                                   depth: str = "moderate") -> Dict[str, Any]:
        """Process a complete research request with clarification"""
        try:
            # Step 1: Analyze request and ask clarifications
            clarification_result = await self.clarification_agent.analyze_research_request(topic, objectives)
            
            # Step 2: Create research query
            query_result = await self.tools.create_research_query(
                topic=topic,
                research_question=f"Research on {topic} with objectives: {', '.join(objectives)}",
                depth=depth,
                sources=["web_search", "document_analysis"],
                constraints=["reliable_sources", "recent_information"],
                output_format="comprehensive_report"
            )
            
            if not query_result["success"]:
                return query_result
            
            query_id = query_result["query_id"]
            
            # Step 3: Conduct research
            research_result = await self.deep_research_agent.research_topic(
                self.tools.research_database[query_id]
            )
            
            # Step 4: Quality evaluation
            # Create mock report for evaluation
            mock_report = ResearchReport(
                report_id=f"temp_{query_id}",
                query_id=query_id,
                topic=topic,
                executive_summary=research_result.content[:300],
                key_findings=[],
                detailed_analysis=research_result.content,
                insights_and_recommendations=[],
                data_visualizations=[],
                limitations=["Mock report for evaluation"],
                confidence_level="High",
                completion_percentage=85.0
            )
            
            quality_result = await self.quality_agent.evaluate_research_quality(mock_report)
            
            # Step 5: Generate improvement suggestions
            improvement_result = await self.quality_agent.suggest_improvements(
                quality_result.content, mock_report
            )
            
            return {
                "success": True,
                "query_id": query_id,
                "research_results": {
                    "clarification": {
                        "content": clarification_result.content[:200],
                        "execution_time": clarification_result.execution_time
                    },
                    "research": {
                        "content": research_result.content[:300],
                        "execution_time": research_result.execution_time,
                        "model": research_result.model_used
                    },
                    "quality_evaluation": {
                        "content": quality_result.content[:200],
                        "execution_time": quality_result.execution_time
                    },
                    "improvements": {
                        "content": improvement_result.content[:200],
                        "execution_time": improvement_result.execution_time
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Research request processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def conduct_comprehensive_research(self, topic: str, research_question: str,
                                         sources: List[str] = None) -> Dict[str, Any]:
        """Conduct comprehensive research with multiple sources"""
        try:
            if sources is None:
                sources = ["web_search", "document_analysis", "database_query"]
            
            # Step 1: Create research query
            query_result = await self.tools.create_research_query(
                topic=topic,
                research_question=research_question,
                depth="comprehensive",
                sources=sources,
                constraints=["authoritative_sources", "recent_data", "multiple_perspectives"],
                output_format="detailed_analysis"
            )
            
            if not query_result["success"]:
                return query_result
            
            query_id = query_result["query_id"]
            
            # Step 2: Conduct web search
            web_results = []
            if "web_search" in sources:
                web_search_result = await self.tools.conduct_web_search(
                    f"{topic} {research_question}", max_results=15
                )
                if web_search_result["success"]:
                    web_results = web_search_result["findings"]
            
            # Step 3: Document analysis (mock for demo)
            doc_results = []
            if "document_analysis" in sources:
                # Mock document analysis
                doc_analysis_result = await self.tools.analyze_documents_for_research(
                    ["Sample research document about " + topic], research_question
                )
                if doc_analysis_result["success"]:
                    doc_results = doc_analysis_result["findings"]
            
            # Step 4: Synthesize findings
            all_findings = web_results + doc_results
            synthesis_result = await self.tools.synthesize_research_findings(
                all_findings, research_question
            )
            
            # Step 5: Create comprehensive report
            if synthesis_result["success"]:
                report_result = await self.tools.create_research_report(
                    query_id=query_id,
                    findings=all_findings,
                    synthesis=synthesis_result["synthesized_content"],
                    analysis=synthesis_result["synthesized_content"]
                )
                
                return {
                    "success": True,
                    "query_id": query_id,
                    "comprehensive_results": {
                        "web_search": {
                            "results_count": len(web_results),
                            "findings": web_results[:5]  # Top 5 findings
                        },
                        "document_analysis": {
                            "results_count": len(doc_results),
                            "findings": doc_results[:3]  # Top 3 findings
                        },
                        "synthesis": synthesis_result,
                        "report": report_result
                    }
                }
            else:
                return {"success": False, "error": "Synthesis failed"}
                
        except Exception as e:
            logger.error(f"Comprehensive research failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get deep research system statistics"""
        return {
            "agents": {
                "deep_research_agent": self.deep_research_agent.name,
                "clarification_agent": self.clarification_agent.name,
                "quality_agent": self.quality_agent.name
            },
            "tools": [
                "create_research_query",
                "conduct_web_search",
                "analyze_documents_for_research",
                "synthesize_research_findings",
                "create_research_report",
                "get_research_report"
            ],
            "database_stats": {
                "total_queries": len(self.tools.research_database),
                "total_reports": len(self.tools.reports_database)
            }
        }
