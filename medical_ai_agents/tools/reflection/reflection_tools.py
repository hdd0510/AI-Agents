#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Reflection Tools
---------------------------------
Tools cho intelligent reflection system.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional

from medical_ai_agents.tools.base_tools import BaseTool

class ReflectionAnalysisTool(BaseTool):
    """Tool để analyze agent results và quyết định reflection strategy."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="reflection_analysis",
            description="Phân tích kết quả từ các agents và quyết định chiến lược reflection phù hợp."
        )
    
    def _run(self, agent_results: Dict[str, Any], query: str, 
             image_path: Optional[str] = None, medical_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze agent results để quyết định reflection strategy."""
        
        analysis = {
            "success": True,
            "reflection_needed": False,
            "reflection_strategy": "none",
            "critical_issues": [],
            "confidence_assessment": {},
            "contradiction_detected": False,
            "safety_concerns": [],
            "recommendations": []
        }
        
        try:
            # 1. Analyze individual agent confidence levels
            confidences = self._extract_confidences(agent_results)
            analysis["confidence_assessment"] = confidences
            
            # 2. Detect contradictions between agents
            contradictions = self._detect_contradictions(agent_results)
            analysis["contradiction_detected"] = len(contradictions) > 0
            analysis["contradictions"] = contradictions
            
            # 3. Check for safety concerns
            safety_issues = self._check_safety_concerns(agent_results, query)
            analysis["safety_concerns"] = safety_issues
            
            # 4. Determine if reflection is needed
            reflection_decision = self._decide_reflection_strategy(
                confidences, contradictions, safety_issues, query
            )
            analysis.update(reflection_decision)
            
            return analysis
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "reflection_needed": True,  # Safe fallback
                "reflection_strategy": "safety_review"
            }
    
    def _extract_confidences(self, agent_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence scores từ agent results."""
        confidences = {}
        
        for agent_name, result in agent_results.items():
            if result and result.get("success", False):
                if "confidence" in result:
                    confidences[agent_name] = result["confidence"]
                elif agent_name == "detector_result" and result.get("objects"):
                    # Average confidence của detected objects
                    obj_confs = [obj.get("confidence", 0) for obj in result["objects"]]
                    confidences[agent_name] = sum(obj_confs) / len(obj_confs) if obj_confs else 0
        
        return confidences
    
    def _detect_contradictions(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect contradictions giữa agent results."""
        contradictions = []
        
        # Check detector vs VQA contradiction
        detector = agent_results.get("detector_result", {})
        vqa = agent_results.get("vqa_result", {})
        
        if detector.get("success") and vqa.get("success"):
            det_count = detector.get("count", 0)
            vqa_answer = vqa.get("answer", "").lower()
            
            # Contradiction patterns
            if det_count > 0 and any(phrase in vqa_answer for phrase in ["no polyp", "không có polyp", "not detected"]):
                contradictions.append({
                    "type": "detector_vqa_mismatch",
                    "description": f"Detector phát hiện {det_count} polyp nhưng VQA nói không có",
                    "severity": "high"
                })
            
            elif det_count == 0 and any(phrase in vqa_answer for phrase in ["detected polyp", "tìm thấy polyp", "có polyp"]):
                contradictions.append({
                    "type": "vqa_detector_mismatch", 
                    "description": "VQA báo có polyp nhưng detector không phát hiện",
                    "severity": "high"
                })
        
        # Check classification consistency
        modality = agent_results.get("modality_result", {})
        region = agent_results.get("region_result", {})
        
        if modality.get("success") and region.get("success"):
            mod_conf = modality.get("confidence", 0)
            reg_conf = region.get("confidence", 0)
            
            if abs(mod_conf - reg_conf) > 0.4:
                contradictions.append({
                    "type": "classification_confidence_mismatch",
                    "description": f"Confidence gap lớn: modality={mod_conf:.2f}, region={reg_conf:.2f}",
                    "severity": "medium"
                })
        
        return contradictions
    
    def _check_safety_concerns(self, agent_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Check for medical safety concerns."""
        safety_issues = []
        
        # Low confidence on critical findings
        detector = agent_results.get("detector_result", {})
        if detector.get("success") and detector.get("count", 0) > 0:
            for obj in detector.get("objects", []):
                if obj.get("confidence", 0) < 0.6:
                    safety_issues.append({
                        "type": "low_confidence_detection",
                        "description": f"Polyp detection với confidence thấp: {obj.get('confidence', 0):.2f}",
                        "severity": "high"
                    })
        
        # VQA uncertainty on medical advice
        vqa = agent_results.get("vqa_result", {})
        if vqa.get("success"):
            answer = vqa.get("answer", "").lower()
            uncertainty_phrases = ["không chắc", "có thể", "might be", "unclear", "difficult to determine"]
            
            if any(phrase in answer for phrase in uncertainty_phrases) and "urgent" in query.lower():
                safety_issues.append({
                    "type": "uncertain_urgent_query",
                    "description": "VQA không chắc chắn về câu hỏi khẩn cấp",
                    "severity": "high"
                })
        
        return safety_issues
    
    def _decide_reflection_strategy(self, confidences: Dict, contradictions: List, 
                                  safety_issues: List, query: str) -> Dict[str, Any]:
        """Quyết định reflection strategy dựa trên analysis."""
        
        # Calculate overall confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0
        
        # Decision logic
        if len(safety_issues) > 0:
            return {
                "reflection_needed": True,
                "reflection_strategy": "safety_focused",
                "priority": "high",
                "reason": "Detected safety concerns requiring review"
            }
        
        elif len(contradictions) > 0:
            high_severity = any(c["severity"] == "high" for c in contradictions)
            return {
                "reflection_needed": True,
                "reflection_strategy": "contradiction_resolution" if high_severity else "consistency_check",
                "priority": "high" if high_severity else "medium", 
                "reason": f"Detected {len(contradictions)} contradictions"
            }
        
        elif avg_confidence < 0.5:
            return {
                "reflection_needed": True,
                "reflection_strategy": "confidence_boost",
                "priority": "medium",
                "reason": f"Low average confidence: {avg_confidence:.2f}"
            }
        
        elif avg_confidence < 0.7 and "urgent" in query.lower():
            return {
                "reflection_needed": True,
                "reflection_strategy": "medical_validation",
                "priority": "high",
                "reason": "Medium confidence on urgent medical query"
            }
        
        else:
            return {
                "reflection_needed": False,
                "reflection_strategy": "none",
                "priority": "low",
                "reason": "Results appear consistent and confident"
            }


class ResultSynthesisTool(BaseTool):
    """Tool để synthesize final results từ agent outputs và reflection."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="result_synthesis",
            description="Tổng hợp kết quả cuối cùng từ các agents và reflection analysis."
        )
    
    def _run(self, agent_results: Dict[str, Any], reflection_analysis: Dict[str, Any],
             query: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize final results với reflection insights."""
        
        try:
            synthesis = {
                "success": True,
                "final_answer": "",
                "confidence_score": 0.0,
                "medical_findings": {},
                "safety_status": "validated",
                "reflection_applied": reflection_analysis.get("reflection_needed", False),
                "recommendations": []
            }
            
            # Extract key findings từ agents
            findings = self._extract_medical_findings(agent_results)
            synthesis["medical_findings"] = findings
            
            # Calculate adjusted confidence based on reflection
            base_confidence = self._calculate_base_confidence(agent_results)
            reflection_adjustment = self._get_reflection_adjustment(reflection_analysis)
            final_confidence = max(0.1, min(1.0, base_confidence + reflection_adjustment))
            
            synthesis["confidence_score"] = final_confidence
            
            # Generate final answer
            synthesis["final_answer"] = self._generate_final_answer(
                findings, reflection_analysis, query, final_confidence
            )
            
            # Safety assessment
            if reflection_analysis.get("safety_concerns"):
                synthesis["safety_status"] = "needs_review"
                synthesis["recommendations"].append("Medical professional review recommended")
            
            # Add reflection insights
            if reflection_analysis.get("reflection_needed"):
                synthesis["reflection_insights"] = {
                    "strategy_used": reflection_analysis.get("reflection_strategy"),
                    "issues_addressed": len(reflection_analysis.get("contradictions", [])) + len(reflection_analysis.get("safety_concerns", [])),
                    "confidence_impact": reflection_adjustment
                }
            
            return synthesis
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "final_answer": "Synthesis failed - manual review required",
                "safety_status": "error"
            }
    
    def _extract_medical_findings(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key medical findings từ agent results."""
        findings = {}
        
        # Detection findings
        detector = agent_results.get("detector_result", {})
        if detector.get("success"):
            findings["polyp_detection"] = {
                "count": detector.get("count", 0),
                "objects": detector.get("objects", []),
                "has_findings": detector.get("count", 0) > 0
            }
        
        # Classification findings  
        modality = agent_results.get("modality_result", {})
        if modality.get("success"):
            findings["imaging_technique"] = {
                "type": modality.get("class_name", "unknown"),
                "confidence": modality.get("confidence", 0)
            }
        
        region = agent_results.get("region_result", {})
        if region.get("success"):
            findings["anatomical_region"] = {
                "location": region.get("class_name", "unknown"),
                "confidence": region.get("confidence", 0)
            }
        
        # VQA medical analysis
        vqa = agent_results.get("vqa_result", {})
        if vqa.get("success"):
            findings["medical_analysis"] = {
                "assessment": vqa.get("answer", ""),
                "confidence": vqa.get("confidence", 0)
            }
        
        return findings
    
    def _calculate_base_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate base confidence từ agent results."""
        confidences = []
        
        for result in agent_results.values():
            if result and result.get("success") and "confidence" in result:
                confidences.append(result["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _get_reflection_adjustment(self, reflection_analysis: Dict[str, Any]) -> float:
        """Get confidence adjustment từ reflection analysis."""
        if not reflection_analysis.get("reflection_needed"):
            return 0.0
        
        # Negative adjustment for issues
        adjustment = 0.0
        
        if reflection_analysis.get("safety_concerns"):
            adjustment -= 0.2
        
        if reflection_analysis.get("contradiction_detected"):
            adjustment -= 0.15
        
        strategy = reflection_analysis.get("reflection_strategy", "")
        if strategy == "safety_focused":
            adjustment -= 0.25
        elif strategy == "contradiction_resolution":
            adjustment -= 0.2
        elif strategy == "confidence_boost":
            adjustment += 0.1  # Positive adjustment for clarity
        
        return max(-0.5, min(0.2, adjustment))
    
    def _generate_final_answer(self, findings: Dict, reflection_analysis: Dict, 
                              query: str, confidence: float) -> str:
        """Generate comprehensive final answer."""
        
        parts = []
        
        # Medical findings summary
        polyp_info = findings.get("polyp_detection", {})
        if polyp_info.get("has_findings"):
            count = polyp_info["count"]
            parts.append(f"🔍 **Phát hiện polyp:** Tìm thấy {count} polyp trong hình ảnh nội soi.")
            
            # Add details about polyps
            objects = polyp_info.get("objects", [])
            if objects:
                for i, obj in enumerate(objects[:3]):  # Top 3
                    conf = obj.get("confidence", 0)
                    pos = obj.get("position_description", "unknown position")
                    parts.append(f"- Polyp {i+1}: Độ tin cậy {conf:.1%}, vị trí {pos}")
        else:
            parts.append("🔍 **Kết quả phát hiện:** Không phát hiện polyp trong hình ảnh.")
        
        # Technical details
        imaging = findings.get("imaging_technique", {})
        region = findings.get("anatomical_region", {})
        
        if imaging or region:
            parts.append("\n📋 **Thông tin kỹ thuật:**")
            if imaging:
                parts.append(f"- Kỹ thuật chụp: {imaging.get('type', 'unknown')}")
            if region:
                parts.append(f"- Vùng giải phẫu: {region.get('location', 'unknown')}")
        
        # Medical analysis từ VQA
        medical_analysis = findings.get("medical_analysis", {})
        if medical_analysis:
            assessment = medical_analysis.get("assessment", "")
            if assessment and len(assessment) > 50:
                parts.append(f"\n🏥 **Phân tích y tế:** {assessment}")
        
        # Reflection insights
        if reflection_analysis.get("reflection_needed"):
            parts.append(f"\n🔄 **Đánh giá chất lượng:** Hệ thống đã thực hiện kiểm tra bổ sung.")
            
            if reflection_analysis.get("contradiction_detected"):
                parts.append("⚠️ *Lưu ý: Phát hiện một số mâu thuẫn giữa các phân tích, đã được xem xét.*")
            
            if reflection_analysis.get("safety_concerns"):
                parts.append("🛡️ *Khuyến nghị: Nên tham khảo thêm ý kiến chuyên gia y tế.*")
        
        # Confidence và recommendations
        parts.append(f"\n📊 **Độ tin cậy tổng thể:** {confidence:.1%}")
        
        if confidence < 0.6:
            parts.append("⚠️ **Lưu ý:** Độ tin cậy thấp, nên kiểm tra lại hoặc tham khảo chuyên gia.")
        elif polyp_info.get("has_findings") and confidence > 0.7:
            parts.append("✅ **Khuyến nghị:** Kết quả có độ tin cậy cao, nên tham khảo bác sĩ chuyên khoa.")
        
        return "\n".join(parts)
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "agent_results": {
                "type": "object",
                "description": "Results from all medical AI agents"
            },
            "reflection_analysis": {
                "type": "object", 
                "description": "Analysis results from reflection analysis tool"
            },
            "query": {
                "type": "string",
                "description": "Original user query"
            },
            "image_path": {
                "type": "string",
                "description": "Path to medical image (optional)",
                "required": False
            }
        }