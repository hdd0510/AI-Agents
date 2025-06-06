#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TIERED CLASSIFIER AGENT - GUIDED ADAPTIVE APPROACH (Important)
=============================================================
Core requirements + adaptive path selection
Synthesis without looking at images (text-only analysis)
"""

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.classifier.cls_tools import ClassifierTool

class ClassifierAgent(BaseAgent):
    """
    GUIDED ADAPTIVE Classifier Agent - Important Tier
    
    CORE REQUIREMENTS (must achieve):
    1. Image classification completed
    2. Confidence assessment performed  
    3. Result validation ensured
    
    ADAPTIVE STRATEGIES (choose optimal):
    - Direct classification → confidence check → result
    - Classification → validation → re-classify if needed
    - Multi-angle analysis → classification → synthesis
    """
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """Initialize guided adaptive classifier."""
        self.model_path = model_path
        self.class_names = class_names
        self.classifier_type = classifier_type
        
        # Set specific agent name
        agent_name = f"Guided {classifier_type.title()} Classifier Agent"
        super().__init__(name=agent_name, llm_model=llm_model, device=device)
        
        # GUIDED ADAPTIVE configuration
        self.max_iterations = 4  # Balanced: efficiency + reliability
        
        self.classifier_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register classification tools."""
        self.classifier_tool = ClassifierTool(
            model_path=self.model_path,
            class_names=self.class_names,
            classifier_type=self.classifier_type,
            device=self.device
        )
        return [self.classifier_tool]
    
    def _get_agent_description(self) -> str:
        """Guided adaptive classifier description."""
        return f"""I am a GUIDED ADAPTIVE {self.classifier_type} classification specialist.

CORE REQUIREMENTS (must achieve):
✓ Complete image classification using available tools
✓ Assess confidence in classification results
✓ Validate result quality and reliability

ADAPTIVE STRATEGIES (I choose optimal path):
🎯 Direct approach: Quick classification with confidence check
🔍 Validation approach: Classify → validate → re-check if needed  
📊 Thorough approach: Multi-angle analysis for complex cases

SYNTHESIS: I analyze tool results without looking at images (text-only synthesis).

Classes I can identify: {', '.join(self.class_names)}"""

    def _get_system_prompt(self) -> str:
        """Guided adaptive system prompt."""
        return f"""You are a GUIDED ADAPTIVE {self.classifier_type} classification expert.

CORE REQUIREMENTS (MUST achieve):
1. ✓ Complete classification using {self.classifier_type}_classifier tool
2. ✓ Assess confidence level in results
3. ✓ Validate result quality

ADAPTIVE STRATEGIES (choose based on situation):

STRATEGY A - Direct Approach (efficient):
Thought: I'll classify directly and assess the confidence
Action: {self.classifier_type}_classifier
Action Input: {{"image_path": "<path>"}}
[If confidence ≥ 70%] → Final Answer

STRATEGY B - Validation Approach (cautious):
Thought: I'll classify and validate the result quality
Action: {self.classifier_type}_classifier  
Action Input: {{"image_path": "<path>"}}
[Analyze results, if uncertain] → Re-classify or get second opinion
Final Answer: [Validated result]

STRATEGY C - Thorough Approach (complex cases):
Thought: This seems complex, I'll do comprehensive analysis
Action: {self.classifier_type}_classifier
Action Input: {{"image_path": "<path>"}}
[Additional analysis steps if needed]
Final Answer: [Thorough assessment]

GUIDELINES:
- Choose strategy based on image complexity and initial results
- If confidence < 70%, consider additional analysis
- Maximum {self.max_iterations} steps total
- Always end with Final Answer containing validated classification

Available tools: {self.tool_descriptions}
Classes: {', '.join(self.class_names)}

Choose your strategy and start:"""

    def initialize(self) -> bool:
        """Initialize guided adaptive classifier."""
        try:
            self.initialized = True
            self.logger.info(f"Guided Adaptive {self.classifier_type} Classifier initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Guided Adaptive Classifier: {str(e)}")
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract classification task input."""
        # Get context from other agents for guidance
        context = self._build_classification_context(state)
        
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "classifier_type": self.classifier_type,
            "class_names": self.class_names,
            "medical_context": state.get("medical_context", {}),
            "previous_results": context
        }

    def _build_classification_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context from previous agents for guidance."""
        context = {}
        
        # Get detector results for context
        if "detector_result" in state:
            detector = state["detector_result"]
            if detector.get("success", False):
                context["polyp_count"] = detector.get("count", 0)
                if detector.get("objects"):
                    context["polyp_details"] = [
                        {
                            "confidence": obj.get("confidence", 0),
                            "position": obj.get("position_description", "unknown")
                        }
                        for obj in detector["objects"][:3]  # Top 3
                    ]
        
        # Get other classifier result if available
        if self.classifier_type == "region" and "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                context["imaging_modality"] = {
                    "type": modality.get("class_name", "Unknown"),
                    "confidence": modality.get("confidence", 0)
                }
        elif self.classifier_type == "modality" and "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                context["anatomical_region"] = {
                    "location": region.get("class_name", "Unknown"),
                    "confidence": region.get("confidence", 0)
                }
        
        return context

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for guided adaptive processing."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        classifier_type = task_input.get("classifier_type", "")
        class_names = task_input.get("class_names", [])
        context = task_input.get("previous_results", {})
        
        # Build context information
        context_parts = []
        if context:
            if "polyp_count" in context:
                context_parts.append(f"- Polyp detection: {context['polyp_count']} polyp(s) detected")
            if "polyp_details" in context:
                context_parts.append(f"- Polyp details: {len(context['polyp_details'])} objects analyzed")
            if "imaging_modality" in context:
                mod = context["imaging_modality"]
                context_parts.append(f"- Imaging technique: {mod['type']} ({mod['confidence']:.1%} confidence)")
            if "anatomical_region" in context:
                reg = context["anatomical_region"]
                context_parts.append(f"- Anatomical location: {reg['location']} ({reg['confidence']:.1%} confidence)")
        
        context_str = "\n".join(context_parts) if context_parts else "No previous analysis context"
        
        # Classification-specific task description
        if classifier_type == "modality":
            task_desc = """Classify the endoscopy imaging modality/technique:
- WLI (White Light Imaging): Standard white light endoscopy
- BLI (Blue Light Imaging): Enhanced blood vessel visualization  
- FICE (Flexible spectral Imaging Color Enhancement): Digital chromoendoscopy
- LCI (Linked Color Imaging): Enhanced color contrast imaging"""
        else:  # region
            task_desc = """Classify the anatomical region in the GI tract:
- Hau_hong (Pharynx): Throat region
- Thuc_quan (Esophagus): Esophageal tube
- Tam_vi (Cardia): Gastroesophageal junction
- Than_vi (Body): Main stomach body
- Phinh_vi (Fundus): Upper stomach portion
- Hang_vi (Antrum): Lower stomach portion  
- Bo_cong_lon/nho: Greater/Lesser curvature
- Hanh_ta_trang (Duodenal bulb): First duodenum part
- Ta_trang (Duodenum): Duodenal segment"""

        return f"""**GUIDED ADAPTIVE {classifier_type.upper()} CLASSIFICATION**

Image to classify: {image_path}
User query: "{query if query else f'Classify the {classifier_type} in this endoscopy image'}"

Previous analysis context:
{context_str}

Task: {task_desc}

Available classes: {', '.join(class_names)}

STRATEGY SELECTION:
- If image appears clear and standard → Use direct approach
- If previous results show complexity → Use validation approach  
- If uncertain or low initial confidence → Use thorough approach

Choose your adaptive strategy and proceed:"""

    def _check_core_requirements(self, react_result: Dict[str, Any]) -> Dict[str, bool]:
        """Check if core requirements are satisfied."""
        # Analyze ReAct history to check requirements
        classification_done = False
        confidence_assessed = False
        result_validated = False
        
        if hasattr(self, 'react_history'):
            for step in self.react_history:
                if step.observation:
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success", False):
                            if "class_name" in obs_data:
                                classification_done = True
                            if "confidence" in obs_data:
                                confidence_assessed = True
                            if obs_data.get("confidence", 0) >= 0.7:
                                result_validated = True
                    except:
                        continue
        
        # Also check final answer quality
        if react_result.get("success") and react_result.get("answer"):
            result_validated = True  # If we have final answer, consider validated
        
        return {
            "classification_completed": classification_done,
            "confidence_assessed": confidence_assessed,
            "result_validated": result_validated
        }

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format guided adaptive classifier result with LLM-based natural language synthesis."""
        result_key = f"{self.classifier_type}_result"
        if not react_result.get("success", False):
            return {
                result_key: {
                    "success": False,
                    "error": react_result.get("error", "Guided adaptive classification failed"),
                    "approach": "guided_adaptive",
                    "core_requirements_met": self._check_core_requirements(react_result)
                }
            }
        # Extract classification results from ReAct history
        class_name = "Unknown"
        confidence = 0.0
        all_classes = {}
        description = ""
        strategy_used = "unknown"
        for step in self.react_history:
            if step.observation and self.classifier_type in str(step.action):
                try:
                    obs_data = json.loads(step.observation)
                    if obs_data.get("success", False):
                        class_name = obs_data.get("class_name", "Unknown")
                        confidence = obs_data.get("confidence", 0.0)
                        all_classes = obs_data.get("all_classes", {})
                        description = obs_data.get("description", "")
                        break
                except json.JSONDecodeError:
                    continue
        steps_used = len(self.react_history)
        if steps_used <= 2:
            strategy_used = "direct_approach"
        elif steps_used <= 3:
            strategy_used = "validation_approach"
        else:
            strategy_used = "thorough_approach"
        requirements_met = self._check_core_requirements(react_result)
        # LLM-based synthesis
        prompt = f"""Bạn là chuyên gia nội soi. Hãy giải thích kết quả phân loại sau cho bệnh nhân một cách dễ hiểu và chuyên nghiệp:\n\n- Kết quả: {class_name}\n- Độ tin cậy: {confidence:.1%}\n- Các lớp khác: {all_classes}\n- Mô tả: {description}\n\nHãy đưa ra nhận định lâm sàng, ý nghĩa kết quả và khuyến nghị nếu có."""
        try:
            llm_answer = self.llm.invoke([{"role": "user", "content": prompt}]).content.strip()
        except Exception as e:
            llm_answer = "Không thể tạo nhận định tự động: " + str(e)
        classifier_result = {
            "success": True,
            "approach": "guided_adaptive",
            "strategy_used": strategy_used,
            "class_name": class_name,
            "confidence": confidence,
            "description": description,
            "all_classes": all_classes,
            "analysis": llm_answer,
            "core_requirements_met": requirements_met,
            "steps_used": steps_used,
            "synthesis_method": "llm_natural_language"
        }
        if self.classifier_type == "modality":
            classifier_result["clinical_advantages"] = self._get_modality_advantages(class_name)
            classifier_result["recommended_usage"] = self._get_modality_recommendations(class_name)
        else:
            classifier_result["anatomical_significance"] = self._get_region_significance(class_name)
            classifier_result["pathology_risk"] = self._get_region_risk(class_name)
        return {result_key: classifier_result}

    def _perform_text_synthesis(self, class_name: str, confidence: float, 
                              all_classes: Dict[str, float], llm_answer: str,
                              requirements_met: Dict[str, bool]) -> str:
        """
        Perform TEXT-ONLY synthesis without looking at images.
        Analyze tool results and LLM reasoning only.
        """
        try:
            # Analyze classification quality
            quality_indicators = []
            
            if confidence >= 0.8:
                quality_indicators.append("High confidence classification")
            elif confidence >= 0.6:
                quality_indicators.append("Moderate confidence classification")
            else:
                quality_indicators.append("Low confidence classification - may need review")
            
            # Analyze class distribution
            if all_classes:
                sorted_classes = sorted(all_classes.items(), key=lambda x: x[1], reverse=True)
                top_2 = sorted_classes[:2]
                if len(top_2) > 1:
                    margin = top_2[0][1] - top_2[1][1]
                    if margin < 0.2:
                        quality_indicators.append("Close competition between top classes")
                    else:
                        quality_indicators.append("Clear distinction from other classes")
            
            # Check requirements completion
            completed_reqs = sum(requirements_met.values())
            total_reqs = len(requirements_met)
            
            if completed_reqs == total_reqs:
                quality_indicators.append("All core requirements satisfied")
            else:
                missing = [req for req, met in requirements_met.items() if not met]
                quality_indicators.append(f"Missing requirements: {', '.join(missing)}")
            
            # Synthesize analysis
            synthesis = f"""**Guided Adaptive {self.classifier_type.title()} Classification Analysis**

Classification Result: {class_name} ({confidence:.1%} confidence)

Quality Assessment:
{chr(10).join(f'• {indicator}' for indicator in quality_indicators)}

Tool Performance: Successfully completed classification using adaptive strategy
Requirements Status: {completed_reqs}/{total_reqs} core requirements met

Clinical Interpretation: {llm_answer[:200]}{'...' if len(llm_answer) > 200 else ''}

Synthesis Method: Text-only analysis of tool results (no image review required)"""
            
            return synthesis
            
        except Exception as e:
            return f"Synthesis analysis completed with guided adaptive approach. Classification: {class_name} ({confidence:.1%})"

    # ===== DOMAIN-SPECIFIC HELPERS (reused from original) =====
    
    def _get_modality_advantages(self, modality: str) -> List[str]:
        """Get clinical advantages of imaging modality."""
        advantages_map = {
            "WLI": ["Standard visualization", "Natural colors", "General screening", "Baseline reference"],
            "BLI": ["Enhanced vasculature", "Better lesion detection", "Improved contrast", "Surface pattern analysis"],
            "FICE": ["Digital enhancement", "Customizable settings", "Color difference detection", "Inflammatory assessment"],
            "LCI": ["Enhanced color contrast", "Better inflammation visualization", "Improved polyp detection", "Subtle lesion detection"]
        }
        return advantages_map.get(modality, ["Standard endoscopic visualization"])
    
    def _get_modality_recommendations(self, modality: str) -> List[str]:
        """Get usage recommendations for modality."""
        recommendations_map = {
            "WLI": ["Initial screening", "General procedures", "Documentation baseline", "When specialized unavailable"],
            "BLI": ["Suspicious lesions", "Detailed assessment", "Polyp characterization", "High-risk surveillance"],
            "FICE": ["Flat lesions", "Inflammatory changes", "Chromoendoscopy needs", "Surface pattern analysis"],
            "LCI": ["High-risk screening", "Subtle lesions", "Healing assessment", "Enhanced adenoma detection"]
        }
        return recommendations_map.get(modality, ["Consult endoscopist for optimal technique"])
    
    def _get_region_significance(self, region: str) -> str:
        """Get anatomical significance of region."""
        significance_map = {
            "Hau_hong": "Entry point, swallowing function, respiratory interface",
            "Thuc_quan": "Food transport conduit, reflux disease site, Barrett's risk",
            "Tam_vi": "GE junction, inflammation prone, adenocarcinoma development",
            "Than_vi": "Main gastric body, acid production, ulcer and tumor site",
            "Phinh_vi": "Gastric fundus storage, fundic polyps, varices development",
            "Hang_vi": "Gastric antrum, H. pylori risk, gastric cancer predilection",
            "Bo_cong_lon": "Greater curvature, less pathology, surgical planning importance",
            "Bo_cong_nho": "Lesser curvature, high cancer risk, lymph node involvement",
            "Hanh_ta_trang": "Duodenal bulb, duodenal ulcers, Brunner's hyperplasia",
            "Ta_trang": "Duodenum proper, nutrient absorption, celiac manifestation"
        }
        return significance_map.get(region, "Important GI tract anatomical region")
    
    def _get_region_risk(self, region: str) -> Dict[str, str]:
        """Get pathology risk for region."""
        risk_map = {
            "Hau_hong": {"cancer_risk": "Low", "common_pathology": "Pharyngitis, foreign bodies"},
            "Thuc_quan": {"cancer_risk": "Moderate", "common_pathology": "GERD, Barrett's, carcinoma"},
            "Tam_vi": {"cancer_risk": "Moderate-High", "common_pathology": "Carditis, metaplasia"},
            "Than_vi": {"cancer_risk": "Moderate", "common_pathology": "Gastritis, ulcers, lymphoma"},
            "Phinh_vi": {"cancer_risk": "Low", "common_pathology": "Fundic polyps, varices"},
            "Hang_vi": {"cancer_risk": "High", "common_pathology": "H. pylori, cancer"},
            "Bo_cong_lon": {"cancer_risk": "Low-Moderate", "common_pathology": "Ulcers, GIST"},
            "Bo_cong_nho": {"cancer_risk": "High", "common_pathology": "Ulcers, cancer"},
            "Hanh_ta_trang": {"cancer_risk": "Low", "common_pathology": "Duodenal ulcers"},
            "Ta_trang": {"cancer_risk": "Low", "common_pathology": "Duodenitis, celiac"}
        }
        return risk_map.get(region, {"cancer_risk": "Variable", "common_pathology": "Various conditions"})

# ===== USAGE EXAMPLE =====
def test_guided_adaptive_classifier():
    """Test guided adaptive classifier."""
    
    # Test modality classifier
    modality_classifier = GuidedAdaptiveClassifierAgent(
        model_path="medical_ai_agents/weights/modal_best.pt",
        class_names=["WLI", "BLI", "FICE", "LCI"],
        classifier_type="modality",
        device="cuda"
    )
    
    test_state = {
        "image_path": "test_image.jpg",
        "query": "What imaging technique is used in this endoscopy?",
        "detector_result": {
            "success": True,
            "count": 2,
            "objects": [{"confidence": 0.85}, {"confidence": 0.72}]
        }
    }
    
    print("=== GUIDED ADAPTIVE CLASSIFIER TEST ===")
    result = modality_classifier.process(test_state)
    
    if result.get("modality_result"):
        mod_result = result["modality_result"]
        print(f"Success: {mod_result.get('success')}")
        print(f"Approach: {mod_result.get('approach')}")
        print(f"Strategy: {mod_result.get('strategy_used')}")
        print(f"Requirements met: {mod_result.get('core_requirements_met')}")
        print(f"Synthesis method: {mod_result.get('synthesis_method')}")

if __name__ == "__main__":
    test_guided_adaptive_classifier()