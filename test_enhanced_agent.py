#!/usr/bin/env python3
"""
Test suite for Enhanced Cybersecurity Response Agent
Demonstrates robust schema validation, guardrails, and structured responses
"""

import json
import unittest
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
from pydantic import ValidationError

# Import our enhanced classes
from respone_agent import (
    ThreatDetection, ResponsePlan, EnhancedGeminiResponseAgent,
    ThreatType, ThreatSeverity, ResponseAction, OrganizationPolicy,
    run_enhanced_detection, load_organization_policies
)


class TestThreatDetection(unittest.TestCase):
    """Test Pydantic schema validation for threat detection"""
    
    def test_valid_threat_detection(self):
        """Test valid threat detection creation"""
        threat = ThreatDetection(
            alert=1,
            threat_type=ThreatType.DDOS,
            confidence=0.95,
            source_ip="192.168.1.100",
            destination_port=80,
            target_host="web-server-01"
        )
        
        self.assertEqual(threat.threat_type, ThreatType.DDOS)
        self.assertEqual(threat.confidence, 0.95)
        self.assertEqual(threat.source_ip, "192.168.1.100")
    
    def test_invalid_confidence_range(self):
        """Test that confidence must be between 0 and 1"""
        with self.assertRaises(ValidationError):
            ThreatDetection(
                alert=1,
                threat_type=ThreatType.DDOS,
                confidence=1.5,  # Invalid: > 1.0
                source_ip="192.168.1.100"
            )
    
    def test_invalid_ip_format(self):
        """Test IP address validation"""
        with self.assertRaises(ValidationError):
            ThreatDetection(
                alert=1,
                threat_type=ThreatType.DDOS,
                confidence=0.95,
                source_ip="not.an.ip.address"  # Invalid IP
            )
    
    def test_port_range_validation(self):
        """Test port number validation"""
        with self.assertRaises(ValidationError):
            ThreatDetection(
                alert=1,
                threat_type=ThreatType.PORTSCAN,
                confidence=0.85,
                source_ip="10.0.0.1",
                destination_port=70000  # Invalid: > 65535
            )


class TestResponsePlan(unittest.TestCase):
    """Test response plan schema validation"""
    
    def test_valid_response_plan(self):
        """Test valid response plan creation"""
        plan = ResponsePlan(
            action=ResponseAction.BLOCK_IP,
            target="192.168.1.100",
            priority=ThreatSeverity.HIGH,
            explanation="Detected high-confidence DDoS attack from this IP"
        )
        
        self.assertEqual(plan.action, ResponseAction.BLOCK_IP)
        self.assertEqual(plan.priority, ThreatSeverity.HIGH)
        self.assertGreaterEqual(len(plan.explanation), 10)
    
    def test_explanation_too_short(self):
        """Test that explanation must be meaningful"""
        with self.assertRaises(ValidationError):
            ResponsePlan(
                action=ResponseAction.ALERT_ADMIN,
                target="server-01",
                priority=ThreatSeverity.MEDIUM,
                explanation="Short"  # Too short
            )


class TestEnhancedDetection(unittest.TestCase):
    """Test enhanced detection functionality"""
    
    def test_run_enhanced_detection(self):
        """Test enhanced detection with real data"""
        df = pd.DataFrame([
            {"Label": "BENIGN", "Source IP": "10.0.0.1", "Destination Port": 443},
            {"Label": "DDoS", "Source IP": "203.0.113.45", "Destination Port": 80, "confidence": 0.92},
            {"Label": "PortScan", "Source IP": "192.168.1.55", "Destination Port": 22, "confidence": 0.87}
        ])
        
        detections = run_enhanced_detection(df)
        
        # Should detect 2 threats (excluding BENIGN)
        self.assertEqual(len(detections), 2)
        
        # Check first detection (DDoS)
        ddos_detection = next(d for d in detections if d.threat_type == ThreatType.DDOS)
        self.assertEqual(ddos_detection.source_ip, "203.0.113.45")
        self.assertEqual(ddos_detection.severity, ThreatSeverity.HIGH)
        
        # Check second detection (PortScan)
        portscan_detection = next(d for d in detections if d.threat_type == ThreatType.PORTSCAN)
        self.assertEqual(portscan_detection.source_ip, "192.168.1.55")
        self.assertEqual(portscan_detection.severity, ThreatSeverity.MEDIUM)


class TestGuardrails(unittest.TestCase):
    """Test guardrail functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.policies = [
            OrganizationPolicy(
                name="Test Policy",
                description="Test policy for guardrails",
                default_action=ResponseAction.ALERT_ADMIN,
                escalation_threshold=0.7
            )
        ]
    
    @patch('respone_agent.dspy')
    @patch('respone_agent.genai.Client')
    def test_high_confidence_no_action_escalation(self, mock_client, mock_dspy):
        """Test that high confidence threats don't result in no_action"""
        # Mock DSPy response
        mock_threat_response = Mock()
        mock_threat_response.action = "no_action"
        mock_threat_response.target = "192.168.1.100"
        mock_threat_response.priority = "low"
        mock_threat_response.explanation = "Low risk assessment"
        mock_threat_response.confidence = "0.9"
        
        mock_dspy.ChainOfThought.return_value = lambda **kwargs: mock_threat_response
        mock_dspy.configure = Mock()
        mock_dspy.LM = Mock()
        
        agent = EnhancedGeminiResponseAgent(
            org_policies=self.policies,
            api_key="test_key"
        )
        
        # Create high confidence threat
        threat_data = {
            "alert": 1,
            "threat_type": "DDoS",
            "confidence": 0.95,  # High confidence
            "source_ip": "192.168.1.100",
            "destination_port": 80,
            "timestamp": datetime.now().isoformat(),
            "severity": "high"
        }
        
        response = agent.decide_response(threat_data)
        
        # Should escalate to alert_admin due to high confidence
        self.assertNotEqual(response.action, ResponseAction.NO_ACTION)
        self.assertIn("high confidence threshold", response.explanation)


def run_demo_scenarios():
    """Run demonstration scenarios showing the enhanced capabilities"""
    print("\n" + "="*80)
    print("ENHANCED CYBERSECURITY RESPONSE AGENT DEMONSTRATION")
    print("="*80)
    
    # Scenario 1: Valid threat processing
    print("\n1. VALID THREAT PROCESSING")
    print("-" * 40)
    
    try:
        threat = ThreatDetection(
            alert=1,
            threat_type=ThreatType.INFILTRATION,
            confidence=0.94,
            source_ip="203.0.113.15",
            destination_port=443,
            target_host="critical-server-01",
            severity=ThreatSeverity.CRITICAL
        )
        print(f"✅ Valid threat created: {threat.threat_type.value} from {threat.source_ip}")
        print(f"   Confidence: {threat.confidence}, Severity: {threat.severity.value}")
    except ValidationError as e:
        print(f"❌ Threat validation failed: {e}")
    
    # Scenario 2: Invalid data handling
    print("\n2. INVALID DATA HANDLING")
    print("-" * 40)
    
    try:
        invalid_threat = ThreatDetection(
            alert=1,
            threat_type=ThreatType.DDOS,
            confidence=1.5,  # Invalid: > 1.0
            source_ip="invalid.ip.format"  # Invalid IP
        )
    except ValidationError as e:
        print(f"✅ Properly rejected invalid data: {len(e.errors())} validation errors")
        for error in e.errors()[:2]:  # Show first 2 errors
            print(f"   - {error['loc'][0]}: {error['msg']}")
    
    # Scenario 3: Response plan validation
    print("\n3. RESPONSE PLAN VALIDATION")
    print("-" * 40)
    
    try:
        plan = ResponsePlan(
            action=ResponseAction.ISOLATE_HOST,
            target="critical-server-01",
            priority=ThreatSeverity.CRITICAL,
            explanation="Detected critical infiltration attempt with high confidence. Immediate isolation required to prevent lateral movement.",
            confidence=0.94,
            estimated_impact="Potential data exfiltration and system compromise"
        )
        print(f"✅ Valid response plan: {plan.action.value} for {plan.target}")
        print(f"   Priority: {plan.priority.value}, Confidence: {plan.confidence}")
        print(f"   Explanation: {plan.explanation[:60]}...")
    except ValidationError as e:
        print(f"❌ Response plan validation failed: {e}")
    
    # Scenario 4: Enhanced detection processing
    print("\n4. ENHANCED DETECTION PROCESSING")
    print("-" * 40)
    
    sample_data = pd.DataFrame([
        {"Label": "Infiltration", "Source IP": "198.51.100.10", "Destination Port": 443, "confidence": 0.96},
        {"Label": "PortScan", "Source IP": "192.168.1.55", "Destination Port": 22, "confidence": 0.87},
        {"Label": "InvalidData", "Source IP": "invalid", "Destination Port": 99999, "confidence": 2.0}
    ])
    
    detections = run_enhanced_detection(sample_data)
    print(f"✅ Processed {len(detections)} valid detections out of {len(sample_data)} inputs")
    
    for detection in detections:
        print(f"   - {detection.threat_type.value}: {detection.source_ip} "
              f"(Confidence: {detection.confidence:.2f}, Severity: {detection.severity.value})")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED - All scenarios show enhanced robustness!")
    print("="*80)


if __name__ == "__main__":
    # Run the demonstration
    run_demo_scenarios()
    
    print("\n" + "="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
