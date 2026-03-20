"""
Structured Outputs and Guardrails for AskSpark
Week 2 Lab 3 - Enhanced safety and reliability with structured data
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agents import Agent, Runner, trace, function_tool, input_guardrail, GuardrailFunctionOutput
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field, validator
import os

from ..base_agent import AskSparkAgentBase, AgentResponse
from ..tools import AskSparkTools
from ...core.ai_providers import UnifiedAIClient
from ...config.settings import Config
from ...config.logging import get_logger

logger = get_logger(__name__)

# Structured Output Models
class LeadQualificationResult(BaseModel):
    """Structured lead qualification result"""
    company_name: str = Field(..., description="Name of the company")
    qualification_score: int = Field(..., ge=0, le=10, description="Qualification score 0-10")
    industry_fit: str = Field(..., description="Industry fit assessment")
    company_size_fit: str = Field(..., description="Company size fit assessment")
    tech_readiness: str = Field(..., description="Technology readiness assessment")
    budget_indication: str = Field(..., description="Budget indication")
    decision_makers: List[str] = Field(..., description="List of key decision makers")
    pain_points: List[str] = Field(..., description="Identified pain points")
    ai_adoption_potential: str = Field(..., description="AI adoption potential")
    recommended_next_steps: List[str] = Field(..., description="Recommended next steps")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    
    @validator('qualification_score')
    def validate_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Qualification score must be between 0 and 10')
        return v

class EmailContentAnalysis(BaseModel):
    """Structured email content analysis"""
    subject_line: str = Field(..., description="Email subject line")
    personalization_score: float = Field(..., ge=0.0, le=1.0, description="Personalization score")
    professionalism_score: float = Field(..., ge=0.0, le=1.0, description="Professionalism score")
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="Clarity score")
    call_to_action_strength: str = Field(..., description="Call-to-action strength")
    personalization_elements: List[str] = Field(..., description="Personalization elements used")
    improvement_suggestions: List[str] = Field(..., description="Improvement suggestions")
    spam_risk_score: float = Field(..., ge=0.0, le=1.0, description="Spam risk score")
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")

class ModelComparisonStructured(BaseModel):
    """Structured model comparison result"""
    task_description: str = Field(..., description="Task being analyzed")
    models_analyzed: List[Dict[str, Any]] = Field(..., description="Models analyzed with metrics")
    recommended_model: str = Field(..., description="Recommended model")
    recommendation_reasoning: str = Field(..., description="Reasoning for recommendation")
    cost_efficiency_ranking: List[str] = Field(..., description="Models ranked by cost efficiency")
    performance_ranking: List[str] = Field(..., description="Models ranked by performance")
    best_overall_value: str = Field(..., description="Best overall value model")
    confidence_level: str = Field(..., description="Confidence level in recommendation")
    alternative_options: List[str] = Field(..., description="Alternative model options")

class DocumentInsightStructured(BaseModel):
    """Structured document analysis result"""
    document_type: str = Field(..., description="Type of document analyzed")
    key_topics: List[str] = Field(..., description="Key topics identified")
    main_insights: List[str] = Field(..., description="Main insights extracted")
    data_points: List[Dict[str, Any]] = Field(..., description="Important data points")
    recommendations: List[str] = Field(..., description="Recommendations based on content")
    action_items: List[str] = Field(..., description="Action items identified")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each insight")
    summary_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality of summary")

# Guardrail Classes
class ContentSafetyLevel(Enum):
    """Content safety levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    BLOCKED = "blocked"

@dataclass
class GuardrailResult:
    """Result from guardrail check"""
    passed: bool
    safety_level: ContentSafetyLevel
    confidence: float
    issues: List[str]
    suggestions: List[str]
    blocked_content: Optional[str] = None

class InputGuardrails:
    """Input validation and safety guardrails"""
    
    def __init__(self):
        self.blocked_patterns = [
            r'\b(password|secret|token|key)\b',
            r'\b(hack|exploit|vulnerability)\b',
            r'\b(illegal|fraud|scam)\b',
            r'\b(hate|racist|sexist)\b',
            r'\b(violence|threat|harm)\b'
        ]
        
        self.suspicious_patterns = [
            r'\b(urgent|immediate|act now)\b',
            r'\b(click here|free|winner)\b',
            r'\b(millionaire|get rich quick)\b'
        ]
        
        logger.info("Initialized input guardrails")
    
    def check_content_safety(self, content: str, content_type: str = "general") -> GuardrailResult:
        """Check content safety and appropriateness"""
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Blocked content detected: {pattern}")
                confidence -= 0.5
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Suspicious content detected: {pattern}")
                confidence -= 0.2
        
        # Check content length
        if len(content) > 10000:
            issues.append("Content too long")
            confidence -= 0.1
        
        # Check for personal information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        
        if re.search(email_pattern, content):
            issues.append("Email address detected in content")
            confidence -= 0.3
        
        if re.search(phone_pattern, content):
            issues.append("Phone number detected in content")
            confidence -= 0.3
        
        # Determine safety level
        if confidence >= 0.8:
            safety_level = ContentSafetyLevel.SAFE
        elif confidence >= 0.5:
            safety_level = ContentSafetyLevel.CAUTION
        elif confidence >= 0.2:
            safety_level = ContentSafetyLevel.WARNING
        else:
            safety_level = ContentSafetyLevel.BLOCKED
        
        # Generate suggestions
        if issues:
            suggestions.extend([
                "Review content for inappropriate language",
                "Remove personal information",
                "Ensure content is professional and relevant"
            ])
        
        return GuardrailResult(
            passed=safety_level in [ContentSafetyLevel.SAFE, ContentSafetyLevel.CAUTION],
            safety_level=safety_level,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )
    
    def check_email_content(self, subject: str, body: str) -> GuardrailResult:
        """Check email content for spam and compliance"""
        combined_content = f"{subject} {body}"
        
        # Spam indicators
        spam_indicators = [
            r'\b(free|click|unsubscribe|winner)\b',
            r'\b(100%|guarantee|risk free)\b',
            r'\b(urgent|limited time|act now)\b',
            r'[!]{3,}',  # Multiple exclamation marks
            r'\$[0-9,]+',  # Money amounts
        ]
        
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Check spam indicators
        spam_score = 0
        for pattern in spam_indicators:
            if re.search(pattern, combined_content, re.IGNORECASE):
                spam_score += 1
                issues.append(f"Spam indicator: {pattern}")
        
        # Check subject line
        if len(subject) > 100:
            issues.append("Subject line too long")
            spam_score += 1
        
        if subject.isupper():
            issues.append("Subject line all caps")
            spam_score += 1
        
        # Check body content
        if len(body) < 50:
            issues.append("Email body too short")
            spam_score += 1
        
        # Calculate confidence
        confidence = max(0.1, 1.0 - (spam_score * 0.2))
        
        # Determine safety level
        if spam_score == 0:
            safety_level = ContentSafetyLevel.SAFE
        elif spam_score <= 2:
            safety_level = ContentSafetyLevel.CAUTION
        elif spam_score <= 4:
            safety_level = ContentSafetyLevel.WARNING
        else:
            safety_level = ContentSafetyLevel.BLOCKED
        
        # Generate suggestions
        if spam_score > 0:
            suggestions.extend([
                "Reduce spam-like language",
                "Use more professional subject line",
                "Provide more substantial content"
            ])
        
        return GuardrailResult(
            passed=safety_level in [ContentSafetyLevel.SAFE, ContentSafetyLevel.CAUTION],
            safety_level=safety_level,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

class StructuredOutputAgent(AskSparkAgentBase):
    """Agent for generating structured outputs"""
    
    def __init__(self):
        instructions = """
        You are a structured data specialist. Your task is to:
        1. Generate responses in precise, structured formats
        2. Ensure all required fields are populated
        3. Provide confidence scores for all assessments
        4. Follow data validation rules strictly
        5. Maintain consistency in data formats
        
        Structured Output Requirements:
        - Use exact field names as specified
        - Provide data in correct types (strings, lists, numbers)
        - Include confidence scores where required
        - Validate all numeric ranges
        - Ensure lists are properly formatted
        - Provide detailed reasoning in analysis fields
        
        Quality Standards:
        - All required fields must be populated
        - Confidence scores between 0.0 and 1.0
        - Lists must contain valid, non-empty items
        - Text fields must be descriptive and relevant
        - Numeric values must be within specified ranges
        
        Always validate your output before returning.
        """
        super().__init__("StructuredOutputAgent", instructions)
    
    async def generate_lead_qualification(self, company_info: str, 
                                     research_data: str) -> AgentResponse:
        """Generate structured lead qualification"""
        input_text = f"""
        Generate structured lead qualification for:
        
        Company Information: {company_info}
        Research Data: {research_data}
        
        Please provide a complete LeadQualificationResult with:
        - qualification_score (0-10)
        - industry_fit assessment
        - company_size_fit assessment
        - tech_readiness assessment
        - budget_indication
        - decision_makers list
        - pain_points list
        - ai_adoption_potential
        - recommended_next_steps list
        - confidence_score (0.0-1.0)
        
        Ensure all fields are populated and valid.
        """
        return await self.run(input_text)
    
    async def analyze_email_content(self, subject: str, body: str) -> AgentResponse:
        """Analyze email content with structured output"""
        input_text = f"""
        Analyze email content and provide structured analysis:
        
        Subject: {subject}
        Body: {body}
        
        Generate complete EmailContentAnalysis with:
        - personalization_score (0.0-1.0)
        - professionalism_score (0.0-1.0)
        - clarity_score (0.0-1.0)
        - call_to_action_strength assessment
        - personalization_elements list
        - improvement_suggestions list
        - spam_risk_score (0.0-1.0)
        - overall_quality_score (0.0-1.0)
        
        Provide detailed, actionable analysis.
        """
        return await self.run(input_text)
    
    async def compare_models_structured(self, task_description: str, 
                                    models: List[str]) -> AgentResponse:
        """Generate structured model comparison"""
        input_text = f"""
        Generate structured model comparison for:
        
        Task: {task_description}
        Models to analyze: {', '.join(models)}
        
        Provide complete ModelComparisonStructured with:
        - models_analyzed with detailed metrics
        - recommended_model
        - recommendation_reasoning
        - cost_efficiency_ranking
        - performance_ranking
        - best_overall_value
        - confidence_level
        - alternative_options
        
        Include realistic performance metrics and costs.
        """
        return await self.run(input_text)
    
    async def analyze_document_structured(self, document_content: str, 
                                       document_type: str) -> AgentResponse:
        """Generate structured document analysis"""
        input_text = f"""
        Analyze document and provide structured insights:
        
        Document Type: {document_type}
        Content: {document_content}
        
        Generate complete DocumentInsightStructured with:
        - key_topics list
        - main_insights list
        - data_points with specific information
        - recommendations list
        - action_items list
        - confidence_scores for each insight
        - summary_quality_score (0.0-1.0)
        
        Extract specific, actionable insights with confidence levels.
        """
        return await self.run(input_text)

class GuardrailAgent(AskSparkAgentBase):
    """Agent for implementing safety guardrails"""
    
    def __init__(self):
        instructions = """
        You are a safety and compliance specialist. Your task is to:
        1. Monitor all content for safety and compliance
        2. Identify potential risks and violations
        3. Provide guidance on content improvement
        4. Ensure professional and appropriate communication
        5. Maintain ethical standards in all interactions
        
        Safety Focus Areas:
        - Inappropriate or harmful content
        - Personal information exposure
        - Spam and deceptive practices
        - Legal and compliance issues
        - Professional communication standards
        
        Guardrail Actions:
        - Block clearly inappropriate content
        - Flag suspicious content for review
        - Provide improvement suggestions
        - Document all safety decisions
        - Maintain audit trail of violations
        
        Always prioritize safety and compliance.
        """
        super().__init__("GuardrailAgent", instructions)
        self.input_guardrails = InputGuardrails()
    
    async def check_input_safety(self, content: str, context: str = "") -> AgentResponse:
        """Check input content safety"""
        input_text = f"""
        Review content for safety and compliance:
        
        Content: {content}
        Context: {context}
        
        Provide safety assessment including:
        - Safety level (safe/caution/warning/blocked)
        - Specific issues identified
        - Confidence in assessment
        - Recommendations for improvement
        - Whether content should be allowed
        
        Be thorough but fair in your assessment.
        """
        return await self.run(input_text)
    
    async def validate_email_compliance(self, subject: str, body: str) -> AgentResponse:
        """Validate email compliance and spam risk"""
        input_text = f"""
        Validate email for compliance and spam risk:
        
        Subject: {subject}
        Body: {body}
        
        Provide compliance assessment including:
        - Spam risk level
        - Professionalism score
        - Compliance issues
        - Improvement recommendations
        - Overall approval status
        
        Focus on CAN-SPAM compliance and professional standards.
        """
        return await self.run(input_text)

class StructuredOutputsManager:
    """Manager for structured outputs and guardrails"""
    
    def __init__(self):
        self.structured_agent = StructuredOutputAgent()
        self.guardrail_agent = GuardrailAgent()
        self.input_guardrails = InputGuardrails()
        self.unified_client = UnifiedAIClient()
        
        logger.info("Initialized Structured Outputs Manager")
    
    async def process_lead_with_structured_output(self, company_info: str, 
                                             research_data: str) -> Dict[str, Any]:
        """Process lead with structured output and guardrails"""
        try:
            # Step 1: Check input safety
            safety_check = self.input_guardrails.check_content_safety(
                f"{company_info} {research_data}", "lead_qualification"
            )
            
            if not safety_check.passed:
                return {
                    "success": False,
                    "error": "Content safety check failed",
                    "safety_level": safety_check.safety_level.value,
                    "issues": safety_check.issues
                }
            
            # Step 2: Generate structured qualification
            qualification_result = await self.structured_agent.generate_lead_qualification(
                company_info, research_data
            )
            
            # Step 3: Validate structured output
            try:
                # Parse the response as structured data
                structured_data = self._parse_structured_response(
                    qualification_result.content, 
                    LeadQualificationResult
                )
                
                return {
                    "success": True,
                    "structured_data": structured_data,
                    "raw_response": qualification_result.content,
                    "execution_time": qualification_result.execution_time,
                    "safety_check": {
                        "passed": safety_check.passed,
                        "level": safety_check.safety_level.value,
                        "confidence": safety_check.confidence
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to parse structured output: {str(e)}")
                return {
                    "success": False,
                    "error": f"Structured output parsing failed: {str(e)}",
                    "raw_response": qualification_result.content
                }
                
        except Exception as e:
            logger.error(f"Lead processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def analyze_email_with_guardrails(self, subject: str, 
                                         body: str) -> Dict[str, Any]:
        """Analyze email with guardrails and structured output"""
        try:
            # Step 1: Check email compliance
            compliance_check = self.input_guardrails.check_email_content(subject, body)
            
            # Step 2: Generate structured analysis
            analysis_result = await self.structured_agent.analyze_email_content(subject, body)
            
            # Step 3: Validate structured output
            try:
                structured_data = self._parse_structured_response(
                    analysis_result.content,
                    EmailContentAnalysis
                )
                
                return {
                    "success": True,
                    "structured_analysis": structured_data,
                    "compliance_check": {
                        "passed": compliance_check.passed,
                        "level": compliance_check.safety_level.value,
                        "confidence": compliance_check.confidence,
                        "issues": compliance_check.issues,
                        "suggestions": compliance_check.suggestions
                    },
                    "raw_analysis": analysis_result.content,
                    "execution_time": analysis_result.execution_time
                }
                
            except Exception as e:
                logger.error(f"Failed to parse email analysis: {str(e)}")
                return {
                    "success": False,
                    "error": f"Email analysis parsing failed: {str(e)}",
                    "raw_response": analysis_result.content
                }
                
        except Exception as e:
            logger.error(f"Email analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def compare_models_with_guardrails(self, task_description: str, 
                                         models: List[str]) -> Dict[str, Any]:
        """Compare models with structured output and validation"""
        try:
            # Step 1: Check task safety
            safety_check = self.input_guardrails.check_content_safety(
                task_description, "model_comparison"
            )
            
            if not safety_check.passed:
                return {
                    "success": False,
                    "error": "Task description safety check failed",
                    "safety_level": safety_check.safety_level.value,
                    "issues": safety_check.issues
                }
            
            # Step 2: Generate structured comparison
            comparison_result = await self.structured_agent.compare_models_structured(
                task_description, models
            )
            
            # Step 3: Validate structured output
            try:
                structured_data = self._parse_structured_response(
                    comparison_result.content,
                    ModelComparisonStructured
                )
                
                # Additional validation
                validation_result = self._validate_model_comparison(structured_data)
                
                return {
                    "success": True,
                    "structured_comparison": structured_data,
                    "validation": validation_result,
                    "raw_response": comparison_result.content,
                    "execution_time": comparison_result.execution_time,
                    "safety_check": {
                        "passed": safety_check.passed,
                        "level": safety_check.safety_level.value,
                        "confidence": safety_check.confidence
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to parse model comparison: {str(e)}")
                return {
                    "success": False,
                    "error": f"Model comparison parsing failed: {str(e)}",
                    "raw_response": comparison_result.content
                }
                
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_structured_response(self, response_text: str, 
                                  model_class: BaseModel) -> Dict[str, Any]:
        """Parse structured response into Pydantic model"""
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                return model_class(**json_data).dict()
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse the entire response
        try:
            return model_class.parse_raw(response_text).dict()
        except Exception:
            # Final fallback: create minimal valid structure
            return self._create_fallback_structure(model_class)
    
    def _validate_model_comparison(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model comparison data"""
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["task_description", "models_analyzed", "recommended_model"]
        for field in required_fields:
            if field not in structured_data or not structured_data[field]:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")
        
        # Check model list
        if "models_analyzed" in structured_data:
            models = structured_data["models_analyzed"]
            if not isinstance(models, list) or len(models) == 0:
                validation_result["valid"] = False
                validation_result["issues"].append("Models analyzed must be a non-empty list")
        
        return validation_result
    
    def _create_fallback_structure(self, model_class: BaseModel) -> Dict[str, Any]:
        """Create fallback structure for failed parsing"""
        if model_class == LeadQualificationResult:
            return {
                "company_name": "Unknown",
                "qualification_score": 5,
                "industry_fit": "Unknown",
                "company_size_fit": "Unknown",
                "tech_readiness": "Unknown",
                "budget_indication": "Unknown",
                "decision_makers": [],
                "pain_points": [],
                "ai_adoption_potential": "Unknown",
                "recommended_next_steps": [],
                "confidence_score": 0.5
            }
        elif model_class == EmailContentAnalysis:
            return {
                "subject_line": "Unknown",
                "personalization_score": 0.5,
                "professionalism_score": 0.5,
                "clarity_score": 0.5,
                "call_to_action_strength": "Unknown",
                "personalization_elements": [],
                "improvement_suggestions": [],
                "spam_risk_score": 0.5,
                "overall_quality_score": 0.5
            }
        elif model_class == ModelComparisonStructured:
            return {
                "task_description": "Unknown",
                "models_analyzed": [],
                "recommended_model": "Unknown",
                "recommendation_reasoning": "Parsing failed",
                "cost_efficiency_ranking": [],
                "performance_ranking": [],
                "best_overall_value": "Unknown",
                "confidence_level": "Low",
                "alternative_options": []
            }
        else:
            return {"error": "Unknown model type for fallback"}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "agents": {
                "structured_output_agent": self.structured_agent.name,
                "guardrail_agent": self.guardrail_agent.name
            },
            "guardrails": {
                "blocked_patterns_count": len(self.input_guardrails.blocked_patterns),
                "suspicious_patterns_count": len(self.input_guardrails.suspicious_patterns)
            },
            "structured_models": [
                "LeadQualificationResult",
                "EmailContentAnalysis", 
                "ModelComparisonStructured",
                "DocumentInsightStructured"
            ]
        }
