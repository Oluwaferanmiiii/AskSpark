"""
Structured Outputs and Guardrails Demo
Week 2 Lab 3 - Enhanced safety and reliability demonstration
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
notebook_dir = Path().absolute()
askspark_root = notebook_dir.parent.parent
src_path = str(askspark_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .structured_outputs import (
    StructuredOutputsManager,
    StructuredOutputAgent,
    GuardrailAgent,
    InputGuardrails,
    LeadQualificationResult,
    EmailContentAnalysis,
    ModelComparisonStructured,
    DocumentInsightStructured,
    ContentSafetyLevel
)
from .base_agent import AgentManager
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

class StructuredOutputsDemo:
    """Complete demonstration of structured outputs and guardrails"""
    
    def __init__(self):
        self.manager = StructuredOutputsManager()
        self.agent_manager = AgentManager()
        
    async def demo_guardrails(self) -> Dict[str, Any]:
        """Demonstrate guardrail functionality"""
        print("\n=== Guardrails Demo ===")
        
        try:
            guardrails = InputGuardrails()
            
            # Test safe content
            safe_content = "This is professional business content about AI consultancy."
            safe_result = guardrails.check_content_safety(safe_content, "general")
            
            print(f"Safe Content Test:")
            print(f"  Content: {safe_content}")
            print(f"  Passed: {safe_result.passed}")
            print(f"  Safety Level: {safe_result.safety_level.value}")
            print(f"  Confidence: {safe_result.confidence:.2f}")
            
            # Test suspicious content
            suspicious_content = "URGENT! ACT NOW! FREE MONEY! CLICK HERE!"
            suspicious_result = guardrails.check_content_safety(suspicious_content, "general")
            
            print(f"\nSuspicious Content Test:")
            print(f"  Content: {suspicious_content}")
            print(f"  Passed: {suspicious_result.passed}")
            print(f"  Safety Level: {suspicious_result.safety_level.value}")
            print(f"  Issues: {suspicious_result.issues}")
            
            # Test email compliance
            email_subject = "FREE MONEY!!! URGENT ACT NOW"
            email_body = "Click here to get free money now!"
            email_result = guardrails.check_email_content(email_subject, email_body)
            
            print(f"\nEmail Compliance Test:")
            print(f"  Subject: {email_subject}")
            print(f"  Passed: {email_result.passed}")
            print(f"  Safety Level: {email_result.safety_level.value}")
            print(f"  Spam Risk: {1 - email_result.confidence:.2f}")
            
            return {
                "success": True,
                "safe_content": {
                    "passed": safe_result.passed,
                    "level": safe_result.safety_level.value,
                    "confidence": safe_result.confidence
                },
                "suspicious_content": {
                    "passed": suspicious_result.passed,
                    "level": suspicious_result.safety_level.value,
                    "issues": suspicious_result.issues
                },
                "email_compliance": {
                    "passed": email_result.passed,
                    "level": email_result.safety_level.value,
                    "spam_risk": 1 - email_result.confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Guardrails demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_structured_lead_qualification(self) -> Dict[str, Any]:
        """Demonstrate structured lead qualification"""
        print("\n=== Structured Lead Qualification Demo ===")
        
        try:
            # Sample company info
            company_info = """
            TechCorp Inc. is a mid-sized technology company specializing in SaaS solutions.
            They recently raised $10M in Series B funding and are looking to expand their AI capabilities.
            Current revenue: $50M annually, 500 employees.
            """
            
            research_data = """
            TechCorp uses AWS infrastructure and has a basic customer support system.
            Key decision makers: Sarah Johnson (VP Operations), Mike Chen (CTO).
            Pain points: High customer support costs, slow response times, manual processes.
            AI readiness: High - already using basic automation tools.
            Budget: Available for AI consultancy projects.
            """
            
            print(f"Processing lead qualification for TechCorp Inc...")
            
            result = await self.manager.process_lead_with_structured_output(
                company_info, research_data
            )
            
            if result["success"]:
                structured_data = result["structured_data"]
                
                print(f"✓ Structured qualification completed in {result['execution_time']:.2f}s")
                print(f"  Qualification Score: {structured_data['qualification_score']}/10")
                print(f"  Industry Fit: {structured_data['industry_fit']}")
                print(f"  Tech Readiness: {structured_data['tech_readiness']}")
                print(f"  AI Adoption Potential: {structured_data['ai_adoption_potential']}")
                print(f"  Decision Makers: {', '.join(structured_data['decision_makers'])}")
                print(f"  Pain Points: {', '.join(structured_data['pain_points'][:2])}")
                print(f"  Confidence: {structured_data['confidence_score']:.2f}")
                print(f"  Safety Check: {result['safety_check']['level']}")
                
                return {
                    "success": True,
                    "structured_data": structured_data,
                    "execution_time": result["execution_time"],
                    "safety_check": result["safety_check"]
                }
            else:
                print(f"✗ Lead qualification failed: {result['error']}")
                return result
                
        except Exception as e:
            logger.error(f"Lead qualification demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_structured_email_analysis(self) -> Dict[str, Any]:
        """Demonstrate structured email analysis"""
        print("\n=== Structured Email Analysis Demo ===")
        
        try:
            # Sample email
            subject = "AI Consultancy for TechCorp's Customer Support Optimization"
            body = """
            Dear Sarah Johnson,
            
            I noticed TechCorp's recent Series B funding and your focus on customer experience.
            Given your current challenges with support costs and response times, our AI consultancy
            can help reduce costs by 40% while improving customer satisfaction.
            
            Would you be available for a 15-minute call next week to discuss how we've helped
            similar SaaS companies achieve these results?
            
            Best regards,
            Alex Chen
            AskSpark AI Consultancy
            """
            
            print(f"Analyzing email content...")
            
            result = await self.manager.analyze_email_with_guardrails(subject, body)
            
            if result["success"]:
                analysis = result["structured_analysis"]
                compliance = result["compliance_check"]
                
                print(f"✓ Email analysis completed in {result['execution_time']:.2f}s")
                print(f"  Personalization Score: {analysis['personalization_score']:.2f}")
                print(f"  Professionalism Score: {analysis['professionalism_score']:.2f}")
                print(f"  Clarity Score: {analysis['clarity_score']:.2f}")
                print(f"  Call-to-Action: {analysis['call_to_action_strength']}")
                print(f"  Spam Risk: {analysis['spam_risk_score']:.2f}")
                print(f"  Overall Quality: {analysis['overall_quality_score']:.2f}")
                print(f"  Personalization Elements: {', '.join(analysis['personalization_elements'][:3])}")
                print(f"  Compliance Check: {compliance['level']}")
                
                return {
                    "success": True,
                    "structured_analysis": analysis,
                    "compliance_check": compliance,
                    "execution_time": result["execution_time"]
                }
            else:
                print(f"✗ Email analysis failed: {result['error']}")
                return result
                
        except Exception as e:
            logger.error(f"Email analysis demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_structured_model_comparison(self) -> Dict[str, Any]:
        """Demonstrate structured model comparison"""
        print("\n=== Structured Model Comparison Demo ===")
        
        try:
            task_description = "Generate personalized sales outreach emails for B2B SaaS companies"
            models = ["gpt-4o", "gpt-4o-mini", "claude-3-haiku", "gemini-pro"]
            
            print(f"Comparing models for: {task_description}")
            print(f"Models: {', '.join(models)}")
            
            result = await self.manager.compare_models_with_guardrails(
                task_description, models
            )
            
            if result["success"]:
                comparison = result["structured_comparison"]
                validation = result["validation"]
                
                print(f"✓ Model comparison completed in {result['execution_time']:.2f}s")
                print(f"  Recommended Model: {comparison['recommended_model']}")
                print(f"  Reasoning: {comparison['recommendation_reasoning'][:100]}...")
                print(f"  Best Overall Value: {comparison['best_overall_value']}")
                print(f"  Confidence Level: {comparison['confidence_level']}")
                print(f"  Cost Efficiency Ranking: {' > '.join(comparison['cost_efficiency_ranking'][:3])}")
                print(f"  Performance Ranking: {' > '.join(comparison['performance_ranking'][:3])}")
                print(f"  Validation: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
                print(f"  Safety Check: {result['safety_check']['level']}")
                
                return {
                    "success": True,
                    "structured_comparison": comparison,
                    "validation": validation,
                    "execution_time": result["execution_time"],
                    "safety_check": result["safety_check"]
                }
            else:
                print(f"✗ Model comparison failed: {result['error']}")
                return result
                
        except Exception as e:
            logger.error(f"Model comparison demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_guardrail_blocking(self) -> Dict[str, Any]:
        """Demonstrate guardrail blocking functionality"""
        print("\n=== Guardrail Blocking Demo ===")
        
        try:
            test_cases = [
                {
                    "name": "Safe Content",
                    "content": "Professional business inquiry about AI consultancy services",
                    "expected": "safe"
                },
                {
                    "name": "Suspicious Content",
                    "content": "GET RICH QUICK! FREE MONEY! ACT NOW! CLICK HERE!",
                    "expected": "blocked"
                },
                {
                    "name": "Inappropriate Content",
                    "content": "Content with hate speech and harmful instructions",
                    "expected": "blocked"
                },
                {
                    "name": "Personal Info",
                    "content": "Contact John at john@email.com or 555-123-4567 for secrets",
                    "expected": "caution"
                }
            ]
            
            results = {}
            
            for test_case in test_cases:
                print(f"\nTesting: {test_case['name']}")
                print(f"Content: {test_case['content'][:50]}...")
                
                guardrails = InputGuardrails()
                result = guardrails.check_content_safety(
                    test_case['content'], "general"
                )
                
                print(f"  Result: {result.safety_level.value}")
                print(f"  Passed: {result.passed}")
                print(f"  Issues: {len(result.issues)}")
                
                if result.issues:
                    print(f"  Top Issues: {result.issues[:2]}")
                
                results[test_case['name']] = {
                    "expected": test_case['expected'],
                    "actual": result.safety_level.value,
                    "passed": result.passed,
                    "confidence": result.confidence,
                    "issues_count": len(result.issues)
                }
            
            return {
                "success": True,
                "test_results": results,
                "total_tests": len(test_cases)
            }
            
        except Exception as e:
            logger.error(f"Guardrail blocking demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete structured outputs and guardrails demonstration"""
        print("AskSpark Structured Outputs & Guardrails Demo")
        print("=" * 60)
        print("Week 2 Lab 3: Enhanced safety and reliability")
        
        demo_results = {}
        
        # Demo 1: Guardrails
        demo_results['guardrails'] = await self.demo_guardrails()
        
        # Demo 2: Structured Lead Qualification
        demo_results['lead_qualification'] = await self.demo_structured_lead_qualification()
        
        # Demo 3: Structured Email Analysis
        demo_results['email_analysis'] = await self.demo_structured_email_analysis()
        
        # Demo 4: Structured Model Comparison
        demo_results['model_comparison'] = await self.demo_structured_model_comparison()
        
        # Demo 5: Guardrail Blocking
        demo_results['guardrail_blocking'] = await self.demo_guardrail_blocking()
        
        # Summary
        print(f"\n=== Demo Summary ===")
        successful_demos = sum(1 for result in demo_results.values() if result.get('success', False))
        total_demos = len(demo_results)
        
        print(f"Demos completed: {successful_demos}/{total_demos}")
        print(f"Success rate: {(successful_demos/total_demos)*100:.1f}%")
        
        # System stats
        system_stats = self.manager.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"  Active Agents: {len(system_stats['agents'])}")
        print(f"  Guardrail Patterns: {system_stats['guardrails']['blocked_patterns_count']}")
        print(f"  Structured Models: {len(system_stats['structured_models'])}")
        
        return demo_results

# Standalone demo execution
async def run_structured_outputs_demo():
    """Run standalone structured outputs demonstration"""
    demo = StructuredOutputsDemo()
    return await demo.run_complete_demo()

# Quick test function
def test_structured_outputs_integration():
    """Quick test of structured outputs integration"""
    print("Testing Structured Outputs Integration...")
    
    async def run_test():
        try:
            manager = StructuredOutputsManager()
            
            # Test system initialization
            stats = manager.get_system_stats()
            
            return (
                len(stats['agents']) == 2 and
                stats['guardrails']['blocked_patterns_count'] > 0 and
                len(stats['structured_models']) == 4
            )
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    # Run complete demo
    result = asyncio.run(run_structured_outputs_demo())
    print(f"\nStructured Outputs Demo completed. Results: {json.dumps(result, indent=2)}")
