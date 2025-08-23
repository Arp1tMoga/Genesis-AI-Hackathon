# -*- coding: utf-8 -*-
"""
Cybersecurity Threat Detection + Response Pipeline
-------------------------------------------------
Enhanced with Pydantic schemas and DSPy for robust, predictable responses.

Dependencies:
    pip install pandas google-genai python-dotenv pydantic dspy-ai typing-extensions
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from pydantic import BaseModel, Field, validator, ValidationError
import dspy

# ================================================================
# Pydantic Models for Schema Validation
# ================================================================

class ThreatSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResponseAction(str, Enum):
    BLOCK_IP = "block_ip"
    ISOLATE_HOST = "isolate_host"
    ALERT_ADMIN = "alert_admin"
    SCHEDULE_SCAN = "schedule_scan"
    NO_ACTION = "no_action"
    QUARANTINE = "quarantine"

class ThreatType(str, Enum):
    BENIGN = "BENIGN"
    DDOS = "DDoS"
    PORTSCAN = "PortScan"
    BOT = "Bot"
    INFILTRATION = "Infiltration"
    WEB_ATTACK = "Web Attack"
    BRUTE_FORCE = "Brute Force"
    UNKNOWN = "Unknown"

class ThreatDetection(BaseModel):
    """Schema for threat detection data"""
    alert: int = Field(..., ge=0, le=1, description="Binary alert flag")
    threat_type: ThreatType = Field(..., description="Type of detected threat")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    source_ip: str = Field(..., min_length=7, max_length=45, description="Source IP address")
    destination_port: Optional[int] = Field(None, ge=1, le=65535, description="Destination port")
    target_host: Optional[str] = Field(None, description="Target host identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Detection timestamp")
    severity: ThreatSeverity = Field(default=ThreatSeverity.MEDIUM, description="Threat severity level")
    
    @validator('source_ip')
    def validate_ip(cls, v):
        """Basic IP validation"""
        import re
        ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        if not (re.match(ipv4_pattern, v) or re.match(ipv6_pattern, v) or v == "unknown"):
            raise ValueError(f"Invalid IP address format: {v}")
        return v

class ResponsePlan(BaseModel):
    """Schema for response plan output"""
    action: ResponseAction = Field(..., description="Recommended response action")
    target: str = Field(..., min_length=1, description="Target of the action (IP/host/system)")
    priority: ThreatSeverity = Field(..., description="Priority level of response")
    explanation: str = Field(..., min_length=10, max_length=500, description="Justification for the action")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in the recommendation")
    estimated_impact: Optional[str] = Field(None, description="Estimated business impact")
    
    @validator('explanation')
    def validate_explanation(cls, v):
        """Ensure explanation is meaningful"""
        if len(v.strip()) < 10:
            raise ValueError("Explanation must be at least 10 characters long")
        return v.strip()

class OrganizationPolicy(BaseModel):
    """Schema for organization security policies"""
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    threat_types: List[ThreatType] = Field(default_factory=list, description="Applicable threat types")
    default_action: ResponseAction = Field(default=ResponseAction.ALERT_ADMIN, description="Default action")
    escalation_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold for escalation")
    business_hours_only: bool = Field(default=False, description="Apply only during business hours")

# ================================================================
# Load API key and setup logging
# ================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # loads variables from .env into environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. "
                     "Make sure your .env contains: GEMINI_API_KEY=your_key_here")

# ================================================================
# DSPy Signatures for Structured LLM Interactions
# ================================================================

class ThreatResponseSignature(dspy.Signature):
    """Analyze cybersecurity threat and recommend response action."""
    
    threat_data = dspy.InputField(desc="Threat detection data in JSON format")
    organization_policies = dspy.InputField(desc="Organization security policies and guidelines")
    
    action = dspy.OutputField(desc="Recommended response action from: block_ip, isolate_host, alert_admin, schedule_scan, no_action, quarantine")
    target = dspy.OutputField(desc="Target of the action (IP address, hostname, or system identifier)")
    priority = dspy.OutputField(desc="Priority level: low, medium, high, or critical")
    explanation = dspy.OutputField(desc="Clear explanation of why this action is recommended (10-200 words)")
    confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")

class PolicyValidationSignature(dspy.Signature):
    """Validate if response action complies with organizational policies."""
    
    proposed_action = dspy.InputField(desc="Proposed response action details")
    policies = dspy.InputField(desc="Organizational security policies")
    
    is_compliant = dspy.OutputField(desc="Boolean indicating if action is policy compliant")
    compliance_notes = dspy.OutputField(desc="Notes about policy compliance or violations")

# ================================================================
# Enhanced Gemini Response Agent with Guardrails
# ================================================================
class EnhancedGeminiResponseAgent:
    """Enhanced response agent with Pydantic validation and DSPy structure"""
    
    def __init__(self, org_policies: List[OrganizationPolicy] = None, api_key: str = None):
        if not api_key:
            raise ValueError("API key must be provided for Gemini client")
        
        self.client = genai.Client(api_key=api_key)
        self.org_policies = org_policies or [
            OrganizationPolicy(
                name="Default Security Policy",
                description="Default containment and investigation procedures",
                default_action=ResponseAction.ALERT_ADMIN,
                escalation_threshold=0.7
            )
        ]
        
        # Configure DSPy with Gemini
        self.lm = dspy.LM(
            model="gemini-2.5-flash",
            api_key=api_key,
            max_tokens=1000
        )
        dspy.configure(lm=self.lm)
        
        # Initialize DSPy modules
        self.threat_response = dspy.ChainOfThought(ThreatResponseSignature)
        self.policy_validator = dspy.ChainOfThought(PolicyValidationSignature)
        
        # Response validation rules
        self.validation_rules = {
            ThreatType.DDOS: {
                "max_confidence_for_no_action": 0.3,
                "required_actions": [ResponseAction.BLOCK_IP, ResponseAction.ALERT_ADMIN]
            },
            ThreatType.PORTSCAN: {
                "max_confidence_for_no_action": 0.4,
                "required_actions": [ResponseAction.BLOCK_IP]
            },
            ThreatType.INFILTRATION: {
                "max_confidence_for_no_action": 0.1,
                "required_actions": [ResponseAction.ISOLATE_HOST, ResponseAction.ALERT_ADMIN]
            }
        }

    def _apply_guardrails(self, threat: ThreatDetection, response: ResponsePlan) -> ResponsePlan:
        """Apply business logic guardrails to the response"""
        logger.info(f"Applying guardrails for threat type: {threat.threat_type}")
        
        # Rule 1: High confidence threats should never result in no_action
        if threat.confidence > 0.8 and response.action == ResponseAction.NO_ACTION:
            logger.warning("High confidence threat with no_action - escalating to alert_admin")
            response.action = ResponseAction.ALERT_ADMIN
            response.priority = ThreatSeverity.HIGH
            response.explanation += " [Escalated due to high confidence threshold]"
        
        # Rule 2: Apply threat-specific rules
        if threat.threat_type in self.validation_rules:
            rules = self.validation_rules[threat.threat_type]
            
            if (response.action == ResponseAction.NO_ACTION and 
                threat.confidence > rules.get("max_confidence_for_no_action", 0.2)):
                # Force a more appropriate action
                response.action = rules["required_actions"][0]
                response.explanation += f" [Auto-escalated per {threat.threat_type} policy]"
                
        # Rule 3: Critical threats require immediate action
        if threat.severity == ThreatSeverity.CRITICAL and response.action == ResponseAction.NO_ACTION:
            response.action = ResponseAction.ISOLATE_HOST
            response.priority = ThreatSeverity.CRITICAL
            
        return response

    def decide_response(self, detection_data: Dict[str, Any]) -> ResponsePlan:
        """Enhanced response decision with validation and guardrails"""
        try:
            # Validate input data using Pydantic
            threat = ThreatDetection(**detection_data)
            logger.info(f"Processing threat: {threat.threat_type} from {threat.source_ip}")
            
            # Prepare policy context
            policy_context = "\n".join([
                f"- {policy.name}: {policy.description} (Default: {policy.default_action})"
                for policy in self.org_policies
            ])
            
            # Use DSPy for structured response generation
            response = self.threat_response(
                threat_data=threat.model_dump_json(),
                organization_policies=policy_context
            )
            
            # Create and validate response plan
            response_plan = ResponsePlan(
                action=ResponseAction(response.action.lower().replace(" ", "_")),
                target=response.target or threat.source_ip,
                priority=ThreatSeverity(response.priority.lower()),
                explanation=response.explanation,
                confidence=float(response.confidence) if response.confidence else 0.8
            )
            
            # Apply guardrails
            response_plan = self._apply_guardrails(threat, response_plan)
            
            # Validate against organizational policies
            policy_check = self.policy_validator(
                proposed_action=response_plan.model_dump_json(),
                policies=policy_context
            )
            
            if hasattr(policy_check, 'is_compliant') and not policy_check.is_compliant:
                logger.warning(f"Policy violation detected: {policy_check.compliance_notes}")
                response_plan.explanation += f" [Policy concern: {policy_check.compliance_notes}]"
            
            logger.info(f"Response decision: {response_plan.action} for {response_plan.target}")
            return response_plan
            
        except ValidationError as e:
            logger.error(f"Validation error in threat data: {e}")
            return ResponsePlan(
                action=ResponseAction.ALERT_ADMIN,
                target=detection_data.get("source_ip", "unknown"),
                priority=ThreatSeverity.HIGH,
                explanation=f"Data validation failed: {str(e)}. Manual review required.",
                confidence=0.9
            )
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {e}")
            return ResponsePlan(
                action=ResponseAction.ALERT_ADMIN,
                target=detection_data.get("source_ip", "unknown"),
                priority=ThreatSeverity.HIGH,
                explanation=f"System error during response generation: {str(e)}",
                confidence=0.5
            )

    def batch_process(self, threats: List[Dict[str, Any]]) -> List[ResponsePlan]:
        """Process multiple threats with batch optimization"""
        responses = []
        for threat_data in threats:
            response = self.decide_response(threat_data)
            responses.append(response)
        return responses


# ================================================================
# Enhanced Detection Agent with Pydantic Validation
# ================================================================
def run_enhanced_detection(df: pd.DataFrame) -> List[ThreatDetection]:
    """Enhanced detection with proper data validation"""
    detections = []
    
    for _, row in df.iterrows():
        try:
            if row["Label"] != "BENIGN":
                # Map threat types to enum values
                threat_type_mapping = {
                    "PortScan": ThreatType.PORTSCAN,
                    "DDoS": ThreatType.DDOS,
                    "Bot": ThreatType.BOT,
                    "Infiltration": ThreatType.INFILTRATION,
                    "Web Attack": ThreatType.WEB_ATTACK,
                    "Brute Force": ThreatType.BRUTE_FORCE
                }
                
                threat_type = threat_type_mapping.get(row["Label"], ThreatType.UNKNOWN)
                
                # Determine severity based on threat type and confidence
                severity_map = {
                    ThreatType.DDOS: ThreatSeverity.HIGH,
                    ThreatType.INFILTRATION: ThreatSeverity.CRITICAL,
                    ThreatType.PORTSCAN: ThreatSeverity.MEDIUM,
                    ThreatType.BOT: ThreatSeverity.HIGH,
                    ThreatType.WEB_ATTACK: ThreatSeverity.HIGH,
                    ThreatType.BRUTE_FORCE: ThreatSeverity.HIGH,
                }
                
                detection = ThreatDetection(
                    alert=1,
                    threat_type=threat_type,
                    confidence=min(0.95, max(0.6, float(row.get("confidence", 0.85)))),
                    source_ip=str(row.get("Source IP", "unknown")),
                    destination_port=int(row.get("Destination Port", 80)) if pd.notna(row.get("Destination Port")) else None,
                    target_host=str(row.get("Target Host", "unknown")),
                    severity=severity_map.get(threat_type, ThreatSeverity.MEDIUM)
                )
                detections.append(detection)
                
        except (ValidationError, ValueError) as e:
            logger.warning(f"Skipping invalid detection data: {e}")
            continue
            
    return detections


# ================================================================
# Enhanced Orchestration with Error Handling
# ================================================================
def process_detection_results_enhanced(df: pd.DataFrame, response_agent: EnhancedGeminiResponseAgent):
    """Enhanced processing with proper error handling and validation"""
    try:
        detections = run_enhanced_detection(df)
        logger.info(f"Processing {len(detections)} valid detections")
        
        if not detections:
            logger.info("No threats detected in the provided data")
            return
        
        # Process detections with enhanced agent
        for detection in detections:
            try:
                # Convert Pydantic model to dict for processing
                detection_dict = detection.dict()
                response_plan = response_agent.decide_response(detection_dict)
                
                print("\n" + "="*50)
                print("=== THREAT DETECTED ===")
                print("="*50)
                print(f"Threat Type: {detection.threat_type.value}")
                print(f"Source IP: {detection.source_ip}")
                print(f"Confidence: {detection.confidence:.2f}")
                print(f"Severity: {detection.severity.value}")
                print(f"Timestamp: {detection.timestamp}")
                
                print("\n--- RESPONSE PLAN ---")
                print(f"Action: {response_plan.action.value}")
                print(f"Target: {response_plan.target}")
                print(f"Priority: {response_plan.priority.value}")
                print(f"Confidence: {response_plan.confidence:.2f}")
                print(f"Explanation: {response_plan.explanation}")
                
                if response_plan.estimated_impact:
                    print(f"Estimated Impact: {response_plan.estimated_impact}")
                
            except Exception as e:
                logger.error(f"Error processing detection {detection.source_ip}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Critical error in detection processing: {e}")
        raise


# ================================================================
# Configuration and Policy Management
# ================================================================
def load_organization_policies() -> List[OrganizationPolicy]:
    """Load organization policies from configuration"""
    return [
        OrganizationPolicy(
            name="Critical Infrastructure Protection",
            description="Immediate isolation for critical system threats",
            threat_types=[ThreatType.INFILTRATION, ThreatType.DDOS],
            default_action=ResponseAction.ISOLATE_HOST,
            escalation_threshold=0.7,
            business_hours_only=False
        ),
        OrganizationPolicy(
            name="Network Scanning Policy", 
            description="Block and alert on port scanning activities",
            threat_types=[ThreatType.PORTSCAN],
            default_action=ResponseAction.BLOCK_IP,
            escalation_threshold=0.6,
            business_hours_only=False
        ),
        OrganizationPolicy(
            name="Web Security Policy",
            description="Monitor and contain web-based attacks",
            threat_types=[ThreatType.WEB_ATTACK, ThreatType.BOT],
            default_action=ResponseAction.SCHEDULE_SCAN,
            escalation_threshold=0.8,
            business_hours_only=True
        )
    ]


# ================================================================
# Main Execution with Enhanced Features
# ================================================================
if __name__ == "__main__":
    try:
        # Example dataset with more realistic data
        df = pd.DataFrame([
            {"Label": "BENIGN", "Destination Port": 443, "Source IP": "10.0.0.5", "confidence": 0.95},
            {"Label": "PortScan", "Destination Port": 22, "Source IP": "192.168.1.55", "confidence": 0.87},
            {"Label": "DDoS", "Destination Port": 80, "Source IP": "203.0.113.45", "confidence": 0.92},
            {"Label": "Infiltration", "Destination Port": 443, "Source IP": "198.51.100.10", "confidence": 0.94},
            {"Label": "Bot", "Destination Port": 8080, "Source IP": "172.16.0.100", "confidence": 0.78}
        ])

        # Load organization policies
        org_policies = load_organization_policies()
        
        # Initialize enhanced response agent
        response_agent = EnhancedGeminiResponseAgent(
            org_policies=org_policies,
            api_key=api_key
        )

        # Process detection results with enhanced pipeline
        print("Starting Enhanced Cybersecurity Response Pipeline...")
        print(f"Loaded {len(org_policies)} organizational policies")
        print("-" * 60)
        
        process_detection_results_enhanced(df, response_agent)
        
        print("\n" + "="*60)
        print("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        print(f"Pipeline failed with error: {e}")
        raise