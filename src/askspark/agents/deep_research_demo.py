"""
Deep Research Demo and Integration
Week 2 Lab 4 - Complete deep research automation demonstration
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src to path
notebook_dir = Path().absolute()
askspark_root = notebook_dir.parent.parent
src_path = str(askspark_root / "src")
if src_path.insert(0, src_path) is None:
    sys.path.insert(0, src_path)

from .deep_research import (
    DeepResearchManager,
    DeepResearchAgent,
    ClarificationAgent,
    ResearchQualityAgent,
    DeepResearchTools,
    ResearchQuery,
    ResearchFinding,
    ResearchReport,
    ResearchSource,
    ResearchDepth
)
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

class DeepResearchDemo:
    """Complete demonstration of deep research system"""
    
    def __init__(self):
        self.research_manager = DeepResearchManager()
        
    async def demo_clarification_process(self) -> Dict[str, Any]:
        """Demonstrate research clarification capabilities"""
        print("\n=== Research Clarification Demo ===")
        
        try:
            agent = self.research_manager.clarification_agent
            
            # Sample research requests
            research_requests = [
                {
                    "topic": "AI in healthcare",
                    "objectives": ["market size", "key players", "regulatory challenges"]
                },
                {
                    "topic": "sustainable technology trends",
                    "objectives": ["emerging technologies", "investment patterns", "market adoption"]
                }
            ]
            
            results = {}
            
            for i, request in enumerate(research_requests):
                print(f"\nAnalyzing research request {i+1}: {request['topic']}")
                
                result = await agent.analyze_research_request(
                    request["topic"], request["objectives"]
                )
                
                print(f"✓ Clarification analysis completed in {result.execution_time:.2f}s")
                print(f"  Key questions: {result.content[:200]}...")
                
                results[request["topic"]] = {
                    "clarifications": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used
                }
            
            return {
                "success": True,
                "results": results,
                "total_requests": len(research_requests)
            }
            
        except Exception as e:
            logger.error(f"Clarification demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_deep_research_process(self) -> Dict[str, Any]:
        """Demonstrate deep research capabilities"""
        print("\n=== Deep Research Process Demo ===")
        
        try:
            agent = self.research_manager.deep_research_agent
            
            # Sample research queries
            queries = [
                ResearchQuery(
                    query_id="demo_1",
                    topic="Artificial Intelligence in Financial Services",
                    research_question="What are the current AI applications, market trends, and regulatory considerations in banking?",
                    depth=ResearchDepth.COMPREHENSIVE,
                    sources=[ResearchSource.WEB_SEARCH, ResearchSource.DOCUMENT_ANALYSIS],
                    constraints=["recent_sources", "authoritative_data", "regulatory_compliance"],
                    expected_output_format="comprehensive_analysis"
                ),
                ResearchQuery(
                    query_id="demo_2",
                    topic="Climate Technology Investment Trends",
                    research_question="What are the investment patterns, key technologies, and market opportunities in climate tech?",
                    depth=ResearchDepth.DEEP,
                    sources=[ResearchSource.WEB_SEARCH, ResearchSource.DATABASE_QUERY],
                    constraints=["venture_capital_data", "market_analysis", "technology_trends"],
                    expected_output_format="investment_report"
                )
            ]
            
            results = {}
            
            for query in queries:
                print(f"\nResearching: {query.topic}")
                print(f"Depth: {query.depth.value}")
                print(f"Sources: {[source.value for source in query.sources]}")
                
                result = await agent.research_topic(query)
                
                print(f"✓ Research completed in {result.execution_time:.2f}s")
                print(f"  Key findings: {result.content[:300]}...")
                
                results[query.topic] = {
                    "research_content": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used,
                    "depth": query.depth.value,
                    "sources": [source.value for source in query.sources]
                }
            
            return {
                "success": True,
                "results": results,
                "total_queries": len(queries)
            }
            
        except Exception as e:
            logger.error(f"Deep research demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_research_synthesis(self) -> Dict[str, Any]:
        """Demonstrate research synthesis capabilities"""
        print("\n=== Research Synthesis Demo ===")
        
        try:
            agent = self.research_manager.deep_research_agent
            
            # Sample research findings
            findings = [
                ResearchFinding(
                    finding_id="synth_1",
                    query_id="demo_synthesis",
                    source=ResearchSource.WEB_SEARCH,
                    content="AI in banking is experiencing rapid adoption, with 65% of banks implementing AI solutions by 2025. Key applications include fraud detection, customer service automation, and risk assessment.",
                    confidence_score=0.85,
                    relevance_score=0.9,
                    source_url="https://example.com/ai-banking-report",
                    source_title="AI in Banking Report 2024"
                ),
                ResearchFinding(
                    finding_id="synth_2",
                    query_id="demo_synthesis",
                    source=ResearchSource.DOCUMENT_ANALYSIS,
                    content="Regulatory frameworks are evolving, with EU AI Act and US Executive Orders creating compliance requirements. Banks must ensure transparency, fairness, and accountability in AI systems.",
                    confidence_score=0.9,
                    relevance_score=0.85,
                    extracted_data={"regulation": "EU AI Act", "compliance": "transparency_required"}
                ),
                ResearchFinding(
                    finding_id="synth_3",
                    query_id="demo_synthesis",
                    source=ResearchSource.EXPERT_CONSULTATION,
                    content="Investment in banking AI technology reached $15.2B in 2024, with expected CAGR of 22% through 2030. Focus areas include explainable AI and regulatory compliance.",
                    confidence_score=0.8,
                    relevance_score=0.8,
                    extracted_data={"investment": 15.2, "cagr": 0.22, "focus": "explainable_ai"}
                )
            ]
            
            synthesis_question = "What is the overall state of AI in banking and what are the key recommendations for banks?"
            
            print(f"Synthesizing {len(findings)} research findings...")
            print(f"Question: {synthesis_question}")
            
            result = await agent.synthesize_findings(findings, synthesis_question)
            
            print(f"✓ Synthesis completed in {result.execution_time:.2f}s")
            print(f"  Integrated analysis: {result.content[:400]}...")
            
            return {
                "success": True,
                "synthesis_question": synthesis_question,
                "synthesized_content": result.content,
                "execution_time": result.execution_time,
                "model": result.model_used,
                "sources_synthesized": len(findings)
            }
            
        except Exception as e:
            logger.error(f"Research synthesis demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_quality_evaluation(self) -> Dict[str, Any]:
        """Demonstrate research quality evaluation"""
        print("\n=== Research Quality Evaluation Demo ===")
        
        try:
            agent = self.research_manager.quality_agent
            
            # Sample research report
            sample_report = ResearchReport(
                report_id="quality_demo",
                query_id="demo_quality",
                topic="AI in Healthcare",
                executive_summary="AI is transforming healthcare through diagnostic assistance, drug discovery, and personalized treatment. The market is growing at 28% CAGR with major investments from tech companies and healthcare providers.",
                key_findings=[
                    ResearchFinding(
                        finding_id="qual_1",
                        query_id="demo_quality",
                        source=ResearchSource.WEB_SEARCH,
                        content="FDA AI/ML approvals increased by 300% since 2020, indicating rapid regulatory acceptance.",
                        confidence_score=0.9,
                        relevance_score=0.95
                    )
                ],
                detailed_analysis="The healthcare AI market shows strong growth across diagnostic imaging, drug discovery, and clinical decision support. Key drivers include aging population, physician shortage, and need for cost reduction.",
                insights_and_recommendations=[
                    "Focus on regulatory compliance early in development",
                    "Invest in explainable AI for medical applications",
                    "Partner with healthcare providers for validation",
                    "Address data privacy and security concerns proactively"
                ],
                limitations=["Based on publicly available data", "Rapidly evolving regulatory landscape"],
                confidence_level="High",
                completion_percentage=85.0
            )
            
            print(f"Evaluating research quality for: {sample_report.topic}")
            print(f"Confidence Level: {sample_report.confidence_level}")
            print(f"Completion: {sample_report.completion_percentage}%")
            
            result = await agent.evaluate_research_quality(sample_report)
            
            print(f"✓ Quality evaluation completed in {result.execution_time:.2f}s")
            print(f"  Quality assessment: {result.content[:300]}...")
            
            # Generate improvement suggestions
            improvement_result = await agent.suggest_improvements(result.content, sample_report)
            
            print(f"✓ Improvement suggestions generated in {improvement_result.execution_time:.2f}s")
            print(f"  Key improvements: {improvement_result.content[:200]}...")
            
            return {
                "success": True,
                "quality_evaluation": {
                    "assessment": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used
                },
                "improvement_suggestions": {
                    "suggestions": improvement_result.content,
                    "execution_time": improvement_result.execution_time,
                    "model": improvement_result.model_used
                },
                "original_report": {
                    "topic": sample_report.topic,
                    "confidence": sample_report.confidence_level,
                    "completion": sample_report.completion_percentage
                }
            }
            
        except Exception as e:
            logger.error(f"Quality evaluation demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_comprehensive_research_workflow(self) -> Dict[str, Any]:
        """Demonstrate complete research workflow"""
        print("\n=== Comprehensive Research Workflow Demo ===")
        
        try:
            # Sample comprehensive research request
            topic = "Sustainable Technology Market Analysis"
            research_question = "What are the key trends, investment patterns, and growth opportunities in sustainable technology?"
            objectives = [
                "Market size and growth projections",
                "Key technology segments and innovations",
                "Major investors and funding patterns",
                "Regulatory landscape and incentives",
                "Competitive analysis and market leaders"
            ]
            
            print(f"Starting comprehensive research on: {topic}")
            print(f"Objectives: {len(objectives)} key areas")
            
            # Process through research manager
            workflow_result = await self.research_manager.process_research_request(
                topic, objectives, depth="comprehensive"
            )
            
            if workflow_result["success"]:
                print(f"✓ Comprehensive research workflow completed")
                print(f"  Query ID: {workflow_result['query_id']}")
                
                # Show workflow steps
                research_results = workflow_result["research_results"]
                for step_name, step_data in research_results.items():
                    print(f"  {step_name.replace('_', ' ').title()}: {step_data['execution_time']:.2f}s")
                
                # Also demonstrate comprehensive research
                comp_result = await self.research_manager.conduct_comprehensive_research(
                    topic, research_question
                )
                
                if comp_result["success"]:
                    print(f"✓ Comprehensive research with multiple sources completed")
                    print(f"  Report ID: {comp_result['comprehensive_results']['report']['report_id']}")
                    print(f"  Web findings: {comp_result['comprehensive_results']['web_search']['results_count']}")
                    print(f"  Document findings: {comp_result['comprehensive_results']['document_analysis']['results_count']}")
                
                return {
                    "success": True,
                    "workflow_results": workflow_result,
                    "comprehensive_research": comp_result,
                    "topic": topic,
                    "objectives_count": len(objectives)
                }
            else:
                print(f"✗ Research workflow failed: {workflow_result['error']}")
                return workflow_result
                
        except Exception as e:
            logger.error(f"Comprehensive research workflow demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete deep research demonstration"""
        print("AskSpark Deep Research System Demo")
        print("=" * 60)
        print("Week 2 Lab 4: Advanced research automation")
        
        demo_results = {}
        
        # Demo 1: Research Clarification
        demo_results['clarification'] = await self.demo_clarification_process()
        
        # Demo 2: Deep Research Process
        demo_results['deep_research'] = await self.demo_deep_research_process()
        
        # Demo 3: Research Synthesis
        demo_results['synthesis'] = await self.demo_research_synthesis()
        
        # Demo 4: Quality Evaluation
        demo_results['quality_evaluation'] = await self.demo_quality_evaluation()
        
        # Demo 5: Comprehensive Workflow
        demo_results['comprehensive_workflow'] = await self.demo_comprehensive_research_workflow()
        
        # Summary
        print(f"\n=== Demo Summary ===")
        successful_demos = sum(1 for result in demo_results.values() if result.get('success', False))
        total_demos = len(demo_results)
        
        print(f"Demos completed: {successful_demos}/{total_demos}")
        print(f"Success rate: {(successful_demos/total_demos)*100:.1f}%")
        
        # System stats
        system_stats = self.research_manager.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"  Active Agents: {len(system_stats['agents'])}")
        print(f"  Available Tools: {len(system_stats['tools'])}")
        print(f"  Research Queries: {system_stats['database_stats']['total_queries']}")
        print(f"  Generated Reports: {system_stats['database_stats']['total_reports']}")
        
        return demo_results

# Standalone demo execution
async def run_deep_research_demo():
    """Run standalone deep research demonstration"""
    demo = DeepResearchDemo()
    return await demo.run_complete_demo()

# Quick test function
def test_deep_research_integration():
    """Quick test of deep research integration"""
    print("Testing Deep Research Integration...")
    
    async def run_test():
        try:
            manager = DeepResearchManager()
            
            # Test system initialization
            stats = manager.get_system_stats()
            
            return (
                len(stats['agents']) == 3 and
                len(stats['tools']) == 6 and
                stats['database_stats']['total_queries'] >= 0 and
                stats['database_stats']['total_reports'] >= 0
            )
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    # Run complete demo
    result = asyncio.run(run_deep_research_demo())
    print(f"\nDeep Research Demo completed. Results: {json.dumps(result, indent=2)}")
