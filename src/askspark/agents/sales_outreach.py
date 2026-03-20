"""
Sales Outreach Agent System for AskSpark Consultancy
Week 2 Lab 2 - Multi-agent sales automation with email integration
"""

import asyncio
import logging
import json
import smtplib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum

from agents import Agent, Runner, trace, function_tool, gen_trace_id
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
import os

from ..base_agent import AskSparkAgentBase, AgentResponse
from ..tools import AskSparkTools
from ...core.ai_providers import UnifiedAIClient
from ...config.settings import Config
from ...config.logging import get_logger
from ...notifications.email_client import EmailClient

logger = get_logger(__name__)

class LeadStatus(Enum):
    """Lead status enumeration"""
    NEW = "new"
    RESEARCHED = "researched"
    CONTACTED = "contacted"
    RESPONDED = "responded"
    CONVERTED = "converted"
    CLOSED = "closed"

@dataclass
class Lead:
    """Lead data structure"""
    company: str
    contact_person: str
    email: str
    industry: str
    size: str
    status: LeadStatus = LeadStatus.NEW
    research_notes: str = ""
    personalized_content: str = ""
    contact_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_contacted: Optional[datetime] = None

@dataclass
class EmailTemplate:
    """Email template structure"""
    template_id: str
    name: str
    subject_template: str
    body_template: str
    variables: List[str]
    category: str

@dataclass
class OutreachCampaign:
    """Outreach campaign structure"""
    campaign_id: str
    name: str
    target_industry: str
    target_company_size: str
    template_id: str
    leads: List[Lead] = field(default_factory=list)
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    sent_count: int = 0
    response_count: int = 0

class LeadResearchAgent(AskSparkAgentBase):
    """Agent for researching and qualifying leads"""
    
    def __init__(self):
        instructions = """
        You are a lead research specialist for AskSpark AI consultancy. Your task is to:
        1. Research companies and identify key decision makers
        2. Analyze company needs and AI adoption potential
        3. Qualify leads based on criteria (industry, size, tech readiness)
        4. Gather relevant information for personalized outreach
        
        Research focus areas:
        - Company business model and challenges
        - Current technology stack and AI usage
        - Key decision makers and their roles
        - Recent news or funding announcements
        - Industry trends and competitive landscape
        
        Provide structured research findings with personalization insights.
        """
        super().__init__("LeadResearchAgent", instructions)
    
    async def research_company(self, company_name: str, industry: str = "") -> AgentResponse:
        """Research a specific company for lead qualification"""
        input_text = f"""
        Research company: {company_name}
        Industry: {industry if industry else 'Not specified'}
        
        Please provide:
        1. Company overview and business model
        2. Key decision makers (CEO, CTO, CIO, etc.)
        3. Current technology and AI adoption level
        4. Potential AI consultancy needs
        5. Personalization insights for outreach
        6. Lead qualification score (1-10)
        
        Format as structured research report.
        """
        return await self.run(input_text)

class EmailPersonalizationAgent(AskSparkAgentBase):
    """Agent for creating personalized email content"""
    
    def __init__(self):
        instructions = """
        You are an expert email personalization specialist for B2B AI consultancy outreach.
        Your task is to create highly personalized email content that:
        1. References specific company research insights
        2. Addresses relevant pain points and challenges
        3. Demonstrates understanding of their industry
        4. Provides clear value proposition for AI consultancy
        5. Includes specific, actionable next steps
        
        Personalization requirements:
        - Use company-specific details and research findings
        - Reference recent company news or achievements
        - Address industry-specific challenges
        - Mention relevant AskSpark case studies or success stories
        - Keep tone professional but conversational
        - Focus on benefits and outcomes, not just features
        
        Create compelling, personalized content that stands out in crowded inboxes.
        """
        super().__init__("EmailPersonalizationAgent", instructions)
    
    async def personalize_email(self, lead: Lead, research_data: str, 
                              template_type: str = "initial_outreach") -> AgentResponse:
        """Create personalized email content for a lead"""
        input_text = f"""
        Create personalized email content for:
        
        Lead Information:
        - Company: {lead.company}
        - Contact: {lead.contact_person}
        - Industry: {lead.industry}
        - Company Size: {lead.size}
        
        Research Data:
        {research_data}
        
        Template Type: {template_type}
        
        Please provide:
        1. Personalized subject line
        2. Email body (3-4 paragraphs max)
        3. Specific call-to-action
        4. Personalization elements used
        
        Make it compelling and specific to this company.
        """
        return await self.run(input_text)

class EmailDeliveryAgent(AskSparkAgentBase):
    """Agent for managing email delivery and tracking"""
    
    def __init__(self):
        instructions = """
        You are an email delivery and tracking specialist. Your task is to:
        1. Optimize email delivery timing and frequency
        2. Track email engagement metrics
        3. Manage follow-up sequences
        4. Handle deliverability issues
        5. Coordinate with other outreach agents
        
        Focus on:
        - Best send times based on recipient timezone and industry
        - A/B testing subject lines and content
        - Follow-up scheduling and cadence
        - Response rate optimization
        - Spam prevention and deliverability
        
        Ensure professional, timely, and effective email delivery.
        """
        super().__init__("EmailDeliveryAgent", instructions)
        self.email_client = EmailClient()
    
    async def optimize_send_time(self, lead: Lead, campaign_data: Dict) -> AgentResponse:
        """Determine optimal send time for email"""
        input_text = f"""
        Determine optimal email send time for:
        
        Lead: {lead.company} ({lead.industry})
        Campaign: {campaign_data.get('name', 'General outreach')}
        Timezone: Assume business hours (9AM-5PM local time)
        
        Consider:
        - Industry best practices
        - Day of week preferences
        - Business hours timing
        - Campaign urgency
        - Previous engagement data
        
        Provide specific send time recommendation with reasoning.
        """
        return await self.run(input_text)
    
    async def plan_followup_sequence(self, lead: Lead, initial_email_sent: datetime) -> AgentResponse:
        """Plan follow-up email sequence"""
        input_text = f"""
        Plan follow-up sequence for:
        
        Lead: {lead.company}
        Initial email sent: {initial_email_sent}
        Industry: {lead.industry}
        
        Provide follow-up plan with:
        1. Number of follow-ups (2-3 recommended)
        2. Timing for each follow-up
        3. Content strategy for each
        4. Escalation points
        5. Stop criteria
        
        Focus on persistence without being annoying.
        """
        return await self.run(input_text)

class SalesOutreachTools:
    """Tools for sales outreach automation"""
    
    def __init__(self):
        self.email_client = EmailClient()
        self.tools = AskSparkTools()
        self.leads_database = {}  # In-memory for demo
        self.campaigns_database = {}
        logger.info("Initialized Sales Outreach Tools")
    
    @function_tool
    async def create_lead(self, company: str, contact_person: str, email: str, 
                        industry: str, size: str, research_notes: str = "") -> Dict[str, Any]:
        """Create a new lead in the system"""
        try:
            lead = Lead(
                company=company,
                contact_person=contact_person,
                email=email,
                industry=industry,
                size=size,
                research_notes=research_notes
            )
            
            # Store in database (simplified)
            lead_id = f"lead_{len(self.leads_database) + 1}"
            self.leads_database[lead_id] = lead
            
            return {
                "success": True,
                "lead_id": lead_id,
                "lead": {
                    "company": lead.company,
                    "contact_person": lead.contact_person,
                    "email": lead.email,
                    "industry": lead.industry,
                    "size": lead.size,
                    "status": lead.status.value,
                    "created_at": lead.created_at.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Failed to create lead: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def get_lead(self, lead_id: str) -> Dict[str, Any]:
        """Retrieve lead information"""
        try:
            lead = self.leads_database.get(lead_id)
            if not lead:
                return {"success": False, "error": "Lead not found"}
            
            return {
                "success": True,
                "lead": {
                    "lead_id": lead_id,
                    "company": lead.company,
                    "contact_person": lead.contact_person,
                    "email": lead.email,
                    "industry": lead.industry,
                    "size": lead.size,
                    "status": lead.status.value,
                    "research_notes": lead.research_notes,
                    "personalized_content": lead.personalized_content,
                    "contact_history": lead.contact_history,
                    "created_at": lead.created_at.isoformat(),
                    "last_contacted": lead.last_contacted.isoformat() if lead.last_contacted else None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get lead: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def send_personalized_email(self, lead_id: str, subject: str, 
                                    body: str, template_type: str = "initial") -> Dict[str, Any]:
        """Send personalized email to lead"""
        try:
            lead = self.leads_database.get(lead_id)
            if not lead:
                return {"success": False, "error": "Lead not found"}
            
            # Create email content
            email_content = {
                "to": lead.email,
                "subject": subject,
                "body": body,
                "template_type": template_type
            }
            
            # Send email (mock implementation)
            send_result = await self._send_email_mock(email_content)
            
            if send_result["success"]:
                # Update lead contact history
                contact_record = {
                    "type": "email",
                    "template_type": template_type,
                    "subject": subject,
                    "sent_at": datetime.now().isoformat(),
                    "email_id": send_result["email_id"]
                }
                lead.contact_history.append(contact_record)
                lead.last_contacted = datetime.now()
                lead.status = LeadStatus.CONTACTED
                
                return {
                    "success": True,
                    "email_id": send_result["email_id"],
                    "sent_at": contact_record["sent_at"],
                    "lead_status": lead.status.value
                }
            else:
                return {"success": False, "error": send_result["error"]}
                
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_email_mock(self, email_content: Dict) -> Dict[str, Any]:
        """Mock email sending for demo purposes"""
        # In production, this would use actual email service
        email_id = f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate email sending
        await asyncio.sleep(0.1)  # Simulate network latency
        
        logger.info(f"Mock email sent to {email_content['to']}: {email_content['subject']}")
        
        return {
            "success": True,
            "email_id": email_id,
            "sent_at": datetime.now().isoformat()
        }
    
    @function_tool
    async def create_campaign(self, name: str, target_industry: str, 
                            target_company_size: str, template_id: str) -> Dict[str, Any]:
        """Create a new outreach campaign"""
        try:
            campaign = OutreachCampaign(
                campaign_id=f"campaign_{len(self.campaigns_database) + 1}",
                name=name,
                target_industry=target_industry,
                target_company_size=target_company_size,
                template_id=template_id
            )
            
            self.campaigns_database[campaign.campaign_id] = campaign
            
            return {
                "success": True,
                "campaign_id": campaign.campaign_id,
                "campaign": {
                    "name": campaign.name,
                    "target_industry": campaign.target_industry,
                    "target_company_size": campaign.target_company_size,
                    "template_id": campaign.template_id,
                    "status": campaign.status,
                    "created_at": campaign.created_at.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Failed to create campaign: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @function_tool
    async def get_campaign_stats(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign statistics"""
        try:
            campaign = self.campaigns_database.get(campaign_id)
            if not campaign:
                return {"success": False, "error": "Campaign not found"}
            
            # Calculate statistics
            total_leads = len(campaign.leads)
            contacted_leads = len([l for l in campaign.leads if l.status == LeadStatus.CONTACTED])
            responded_leads = len([l for l in campaign.leads if l.status == LeadStatus.RESPONDED])
            converted_leads = len([l for l in campaign.leads if l.status == LeadStatus.CONVERTED])
            
            response_rate = (responded_leads / contacted_leads * 100) if contacted_leads > 0 else 0
            conversion_rate = (converted_leads / contacted_leads * 100) if contacted_leads > 0 else 0
            
            return {
                "success": True,
                "stats": {
                    "campaign_id": campaign_id,
                    "campaign_name": campaign.name,
                    "total_leads": total_leads,
                    "contacted_leads": contacted_leads,
                    "responded_leads": responded_leads,
                    "converted_leads": converted_leads,
                    "response_rate": round(response_rate, 2),
                    "conversion_rate": round(conversion_rate, 2),
                    "sent_count": campaign.sent_count,
                    "response_count": campaign.response_count
                }
            }
        except Exception as e:
            logger.error(f"Failed to get campaign stats: {str(e)}")
            return {"success": False, "error": str(e)}

class SalesOutreachManager:
    """Manager for coordinating sales outreach agents and workflows"""
    
    def __init__(self):
        self.lead_research_agent = LeadResearchAgent()
        self.email_personalization_agent = EmailPersonalizationAgent()
        self.email_delivery_agent = EmailDeliveryAgent()
        self.tools = SalesOutreachTools()
        self.unified_client = UnifiedAIClient()
        
        logger.info("Initialized Sales Outreach Manager")
    
    async def process_new_lead(self, company: str, contact_person: str, email: str,
                            industry: str, size: str) -> Dict[str, Any]:
        """Process a new lead through the complete outreach workflow"""
        try:
            # Step 1: Create lead
            lead_result = await self.tools.create_lead(
                company=company,
                contact_person=contact_person,
                email=email,
                industry=industry,
                size=size
            )
            
            if not lead_result["success"]:
                return lead_result
            
            lead_id = lead_result["lead_id"]
            
            # Step 2: Research company
            research_result = await self.lead_research_agent.research_company(company, industry)
            
            # Step 3: Personalize email
            personalization_result = await self.email_personalization_agent.personalize_email(
                lead=Lead(company, contact_person, email, industry, size),
                research_data=research_result.content,
                template_type="initial_outreach"
            )
            
            # Step 4: Optimize send time
            send_time_result = await self.email_delivery_agent.optimize_send_time(
                lead=Lead(company, contact_person, email, industry, size),
                campaign_data={"name": "Initial Outreach"}
            )
            
            # Step 5: Plan follow-ups
            followup_result = await self.email_delivery_agent.plan_followup_sequence(
                lead=Lead(company, contact_person, email, industry, size),
                initial_email_sent=datetime.now()
            )
            
            return {
                "success": True,
                "lead_id": lead_id,
                "workflow_results": {
                    "research": {
                        "content": research_result.content[:200] + "...",
                        "execution_time": research_result.execution_time
                    },
                    "personalization": {
                        "content": personalization_result.content[:200] + "...",
                        "execution_time": personalization_result.execution_time
                    },
                    "send_time_optimization": {
                        "recommendation": send_time_result.content[:100] + "...",
                        "execution_time": send_time_result.execution_time
                    },
                    "followup_plan": {
                        "plan": followup_result.content[:200] + "...",
                        "execution_time": followup_result.execution_time
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process new lead: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def send_outreach_email(self, lead_id: str) -> Dict[str, Any]:
        """Send personalized outreach email to lead"""
        try:
            # Get lead information
            lead_result = await self.tools.get_lead(lead_id)
            if not lead_result["success"]:
                return lead_result
            
            lead_data = lead_result["lead"]
            
            # Research company if not already done
            if not lead_data["research_notes"]:
                research_result = await self.lead_research_agent.research_company(
                    lead_data["company"], lead_data["industry"]
                )
                research_data = research_result.content
            else:
                research_data = lead_data["research_notes"]
            
            # Personalize email
            lead = Lead(
                company=lead_data["company"],
                contact_person=lead_data["contact_person"],
                email=lead_data["email"],
                industry=lead_data["industry"],
                size=lead_data["size"]
            )
            
            personalization_result = await self.email_personalization_agent.personalize_email(
                lead=lead,
                research_data=research_data,
                template_type="initial_outreach"
            )
            
            # Parse email content (simplified)
            content_lines = personalization_result.content.split('\n')
            subject = content_lines[0].replace('Subject:', '').strip()
            body = '\n'.join(content_lines[1:]).strip()
            
            # Send email
            send_result = await self.tools.send_personalized_email(
                lead_id=lead_id,
                subject=subject,
                body=body,
                template_type="initial_outreach"
            )
            
            return send_result
            
        except Exception as e:
            logger.error(f"Failed to send outreach email: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get sales outreach system statistics"""
        return {
            "agents": {
                "lead_research_agent": self.lead_research_agent.name,
                "email_personalization_agent": self.email_personalization_agent.name,
                "email_delivery_agent": self.email_delivery_agent.name
            },
            "tools": ["create_lead", "get_lead", "send_personalized_email", 
                     "create_campaign", "get_campaign_stats"],
            "total_leads": len(self.tools.leads_database),
            "total_campaigns": len(self.tools.campaigns_database)
        }
