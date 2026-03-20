"""
Sales Outreach Demo and Integration
Week 2 Lab 2 - Complete sales outreach automation demonstration
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

from .sales_outreach import (
    SalesOutreachManager,
    LeadResearchAgent,
    EmailPersonalizationAgent,
    EmailDeliveryAgent,
    SalesOutreachTools,
    Lead,
    LeadStatus,
    OutreachCampaign
)
from .base_agent import AgentManager
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

class SalesOutreachDemo:
    """Complete demonstration of sales outreach system"""
    
    def __init__(self):
        self.outreach_manager = SalesOutreachManager()
        self.agent_manager = AgentManager()
        
    async def demo_lead_research(self) -> Dict[str, Any]:
        """Demonstrate lead research capabilities"""
        print("\n=== Lead Research Agent Demo ===")
        
        try:
            agent = self.outreach_manager.lead_research_agent
            
            # Research sample companies
            companies = [
                ("TechCorp Inc.", "Technology"),
                ("Global Manufacturing Co.", "Manufacturing"),
                ("Financial Services Ltd.", "Finance")
            ]
            
            results = {}
            
            for company, industry in companies:
                print(f"\nResearching {company} ({industry})...")
                
                result = await agent.research_company(company, industry)
                
                print(f"✓ Research completed in {result.execution_time:.2f}s")
                print(f"  Preview: {result.content[:150]}...")
                
                results[company] = {
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used
                }
            
            return {
                "success": True,
                "results": results,
                "total_companies": len(companies)
            }
            
        except Exception as e:
            logger.error(f"Lead research demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_email_personalization(self) -> Dict[str, Any]:
        """Demonstrate email personalization capabilities"""
        print("\n=== Email Personalization Agent Demo ===")
        
        try:
            agent = self.outreach_manager.email_personalization_agent
            
            # Sample leads
            leads = [
                Lead(
                    company="TechCorp Inc.",
                    contact_person="Sarah Johnson",
                    email="sarah.j@techcorp.com",
                    industry="Technology",
                    size="500-1000"
                ),
                Lead(
                    company="Global Manufacturing Co.",
                    contact_person="Mike Chen",
                    email="m.chen@globalmfg.com",
                    industry="Manufacturing",
                    size="1000-5000"
                )
            ]
            
            # Sample research data
            research_data = """
            TechCorp Inc. is a mid-sized technology company specializing in SaaS solutions.
            Recent funding round of $10M for AI expansion. Looking to implement AI for customer support automation.
            Key decision maker: Sarah Johnson, VP of Operations.
            Current tech stack: AWS, React, PostgreSQL.
            Pain points: High customer support costs, slow response times.
            """
            
            results = {}
            
            for lead in leads:
                print(f"\nPersonalizing email for {lead.contact_person} at {lead.company}...")
                
                result = await agent.personalize_email(
                    lead=lead,
                    research_data=research_data,
                    template_type="initial_outreach"
                )
                
                print(f"✓ Personalization completed in {result.execution_time:.2f}s")
                print(f"  Preview: {result.content[:200]}...")
                
                results[lead.company] = {
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "model": result.model_used
                }
            
            return {
                "success": True,
                "results": results,
                "total_leads": len(leads)
            }
            
        except Exception as e:
            logger.error(f"Email personalization demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_email_delivery_optimization(self) -> Dict[str, Any]:
        """Demonstrate email delivery optimization"""
        print("\n=== Email Delivery Agent Demo ===")
        
        try:
            agent = self.outreach_manager.email_delivery_agent
            
            # Sample leads
            leads = [
                Lead(
                    company="TechCorp Inc.",
                    contact_person="Sarah Johnson",
                    email="sarah.j@techcorp.com",
                    industry="Technology",
                    size="500-1000"
                ),
                Lead(
                    company="Global Manufacturing Co.",
                    contact_person="Mike Chen",
                    email="m.chen@globalmfg.com",
                    industry="Manufacturing",
                    size="1000-5000"
                )
            ]
            
            results = {}
            
            for lead in leads:
                print(f"\nOptimizing delivery for {lead.company}...")
                
                # Send time optimization
                send_time_result = await agent.optimize_send_time(
                    lead=lead,
                    campaign_data={"name": "AI Consultancy Outreach"}
                )
                
                print(f"✓ Send time optimization: {send_time_result.content[:100]}...")
                
                # Follow-up planning
                followup_result = await agent.plan_followup_sequence(
                    lead=lead,
                    initial_email_sent=datetime.now()
                )
                
                print(f"✓ Follow-up sequence planned: {followup_result.content[:100]}...")
                
                results[lead.company] = {
                    "send_time_optimization": {
                        "content": send_time_result.content,
                        "execution_time": send_time_result.execution_time
                    },
                    "followup_planning": {
                        "content": followup_result.content,
                        "execution_time": followup_result.execution_time
                    }
                }
            
            return {
                "success": True,
                "results": results,
                "total_leads": len(leads)
            }
            
        except Exception as e:
            logger.error(f"Email delivery demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_complete_workflow(self) -> Dict[str, Any]:
        """Demonstrate complete sales outreach workflow"""
        print("\n=== Complete Sales Outreach Workflow Demo ===")
        
        try:
            # Sample new leads
            new_leads = [
                {
                    "company": "InnovateTech Solutions",
                    "contact_person": "David Kim",
                    "email": "david.k@innovatetech.com",
                    "industry": "Technology",
                    "size": "100-500"
                },
                {
                    "company": "Smart Manufacturing LLC",
                    "contact_person": "Lisa Rodriguez",
                    "email": "lisa.r@smartmfg.com",
                    "industry": "Manufacturing",
                    "size": "500-1000"
                }
            ]
            
            results = {}
            
            for lead_data in new_leads:
                print(f"\n--- Processing Lead: {lead_data['company']} ---")
                
                # Process through complete workflow
                workflow_result = await self.outreach_manager.process_new_lead(**lead_data)
                
                if workflow_result["success"]:
                    print(f"✓ Lead processed successfully")
                    print(f"  Lead ID: {workflow_result['lead_id']}")
                    
                    # Show workflow steps
                    workflow_steps = workflow_result["workflow_results"]
                    for step_name, step_data in workflow_steps.items():
                        print(f"  {step_name.replace('_', ' ').title()}: {step_data['execution_time']:.2f}s")
                    
                    # Send the actual email
                    send_result = await self.outreach_manager.send_outreach_email(
                        workflow_result["lead_id"]
                    )
                    
                    if send_result["success"]:
                        print(f"✓ Email sent successfully")
                        print(f"  Email ID: {send_result['email_id']}")
                        print(f"  Sent at: {send_result['sent_at']}")
                    else:
                        print(f"✗ Email send failed: {send_result['error']}")
                    
                    results[lead_data["company"]] = workflow_result
                else:
                    print(f"✗ Lead processing failed: {workflow_result['error']}")
                    results[lead_data["company"]] = workflow_result
            
            return {
                "success": True,
                "results": results,
                "total_leads": len(new_leads)
            }
            
        except Exception as e:
            logger.error(f"Complete workflow demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def demo_campaign_management(self) -> Dict[str, Any]:
        """Demonstrate campaign management capabilities"""
        print("\n=== Campaign Management Demo ===")
        
        try:
            tools = self.outreach_manager.tools
            
            # Create sample campaigns
            campaigns = [
                {
                    "name": "Q1 Technology Outreach",
                    "target_industry": "Technology",
                    "target_company_size": "100-1000",
                    "template_id": "tech_initial_outreach"
                },
                {
                    "name": "Manufacturing AI Adoption",
                    "target_industry": "Manufacturing",
                    "target_company_size": "500-5000",
                    "template_id": "mfg_ai_consultancy"
                }
            ]
            
            created_campaigns = {}
            
            for campaign_data in campaigns:
                print(f"\nCreating campaign: {campaign_data['name']}")
                
                result = await tools.create_campaign(**campaign_data)
                
                if result["success"]:
                    print(f"✓ Campaign created: {result['campaign_id']}")
                    created_campaigns[campaign_data["name"]] = result
                else:
                    print(f"✗ Campaign creation failed: {result['error']}")
            
            # Get campaign statistics
            print(f"\nCampaign Statistics:")
            for campaign_name, campaign_result in created_campaigns.items():
                campaign_id = campaign_result["campaign_id"]
                stats_result = await tools.get_campaign_stats(campaign_id)
                
                if stats_result["success"]:
                    stats = stats_result["stats"]
                    print(f"\n{campaign_name}:")
                    print(f"  Total Leads: {stats['total_leads']}")
                    print(f"  Contacted: {stats['contacted_leads']}")
                    print(f"  Response Rate: {stats['response_rate']}%")
                    print(f"  Conversion Rate: {stats['conversion_rate']}%")
            
            return {
                "success": True,
                "created_campaigns": created_campaigns,
                "total_campaigns": len(created_campaigns)
            }
            
        except Exception as e:
            logger.error(f"Campaign management demo failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete sales outreach demonstration"""
        print("AskSpark Sales Outreach System Demo")
        print("=" * 60)
        print("Week 2 Lab 2: Multi-agent sales automation")
        
        demo_results = {}
        
        # Demo 1: Lead Research
        demo_results['lead_research'] = await self.demo_lead_research()
        
        # Demo 2: Email Personalization
        demo_results['email_personalization'] = await self.demo_email_personalization()
        
        # Demo 3: Email Delivery Optimization
        demo_results['email_delivery'] = await self.demo_email_delivery_optimization()
        
        # Demo 4: Complete Workflow
        demo_results['complete_workflow'] = await self.demo_complete_workflow()
        
        # Demo 5: Campaign Management
        demo_results['campaign_management'] = await self.demo_campaign_management()
        
        # Summary
        print(f"\n=== Demo Summary ===")
        successful_demos = sum(1 for result in demo_results.values() if result.get('success', False))
        total_demos = len(demo_results)
        
        print(f"Demos completed: {successful_demos}/{total_demos}")
        print(f"Success rate: {(successful_demos/total_demos)*100:.1f}%")
        
        # System stats
        system_stats = self.outreach_manager.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"  Active Agents: {len(system_stats['agents'])}")
        print(f"  Available Tools: {len(system_stats['tools'])}")
        print(f"  Leads in System: {system_stats['total_leads']}")
        print(f"  Active Campaigns: {system_stats['total_campaigns']}")
        
        return demo_results

# Standalone demo execution
async def run_sales_outreach_demo():
    """Run standalone sales outreach demonstration"""
    demo = SalesOutreachDemo()
    return await demo.run_complete_demo()

# Quick test function
def test_sales_outreach_integration():
    """Quick test of sales outreach integration"""
    print("Testing Sales Outreach Integration...")
    
    async def run_test():
        try:
            manager = SalesOutreachManager()
            
            # Test system initialization
            stats = manager.get_system_stats()
            
            return (
                len(stats['agents']) == 3 and
                len(stats['tools']) == 5 and
                stats['total_leads'] >= 0 and
                stats['total_campaigns'] >= 0
            )
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    # Run the complete demo
    result = asyncio.run(run_sales_outreach_demo())
    print(f"\nSales Outreach Demo completed. Results: {json.dumps(result, indent=2)}")
