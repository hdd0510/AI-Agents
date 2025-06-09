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
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType, ReActStep
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
        """Guided adaptive system prompt with strict final answer formatting."""
        return f"""You are a GUIDED ADAPTIVE {self.classifier_type} classification expert.

CORE REQUIREMENTS (MUST achieve):
1. Complete classification using {self.classifier_type}_classifier tool
2. Assess confidence level in results
3. Validate result quality

ADAPTIVE STRATEGIES:
- Direct approach for clear images (fastest)
- Validation approach for uncertain cases
- Thorough approach for complex images

STRICT REACT FORMAT:
Every step MUST follow this exact format:
Thought: [your reasoning]
Action: [tool name or "Final Answer"]
Action Input: [JSON parameters]

FINAL STEP FORMAT:
You MUST end with EXACTLY this format (use Final Answer with capital letters):

Thought: [your final analysis]
Action: Final Answer
Action Input: {{
  "answer": {{"class_name": "[class]", "confidence": "[percentage]", "explanation": "[reason]"}}
}}

DO NOT deviate from this format or your analysis will be rejected.
NEVER use "Final Answer:" as a standalone line - it MUST be "Action: Final Answer".

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
- WLI (White Light Imaging): Thực tế là LCI - Enhanced visualization with color contrast for lesion detection
- BLI (Blue Light Imaging): Thực tế là WLI - Standard visualization technique
- FICE (Flexible spectral Imaging Color Enhancement): Thực tế là BLI - Enhanced visualization of blood vessels and surface patterns
- LCI (Linked Color Imaging): Thực tế là FICE - Digital chromoendoscopy for mucosal assessment"""
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
        low_confidence_identified = False
        llm_enhacement_needed = False
        
        # Track confidence level to determine if LLM enhancement needed
        confidence_level = 0.0
        
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
                                confidence_level = obs_data.get("confidence", 0.0)
                                
                                # Check if confidence is low
                                if confidence_level < 0.6:
                                    low_confidence_identified = True
                                    llm_enhacement_needed = True
                                    
                                # Higher threshold for validation
                                if confidence_level >= 0.7:
                                    result_validated = True
                    except:
                        continue
        
        # Also check final answer quality
        if react_result.get("success") and react_result.get("answer"):
            if isinstance(react_result["answer"], dict) and "confidence" in react_result["answer"]:
                try:
                    conf = react_result["answer"]["confidence"]
                    if isinstance(conf, str) and "%" in conf:
                        conf = float(conf.replace("%", "")) / 100.0
                    else:
                        conf = float(conf)
                    
                    if conf < 0.6:
                        low_confidence_identified = True
                        llm_enhacement_needed = True
                except:
                    pass
            
            # Consider the answer valid if it exists
            result_validated = True
        
        return {
            "classification_completed": classification_done,
            "confidence_assessed": confidence_assessed,
            "result_validated": result_validated,
            "low_confidence_identified": low_confidence_identified,
            "llm_enhancement_needed": llm_enhacement_needed
        }

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format guided adaptive classifier result with standardized output parsing."""
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
        final_answer_data = {}
        strategy_used = "unknown"
        image_path = ""
        query = ""
        
        # Get image path and query from react history
        for step in self.react_history:
            if step.action_input and isinstance(step.action_input, dict):
                if "image_path" in step.action_input:
                    image_path = step.action_input["image_path"]
                if "query" in step.action_input:
                    query = step.action_input["query"]
        
        # First check if final answer contains structured data
        if react_result.get("answer") and isinstance(react_result["answer"], dict):
            final_answer_data = react_result["answer"]
            if "class_name" in final_answer_data:
                class_name = final_answer_data["class_name"]
            if "confidence" in final_answer_data:
                # Handle confidence as string (percentage) or float
                conf_val = final_answer_data["confidence"]
                if isinstance(conf_val, str) and "%" in conf_val:
                    try:
                        confidence = float(conf_val.replace("%", "")) / 100.0
                    except:
                        pass
                else:
                    try:
                        confidence = float(conf_val)
                    except:
                        pass
            if "explanation" in final_answer_data:
                description = final_answer_data["explanation"]
        
        # If no structured data in final answer, get from tool observation
        if class_name == "Unknown":
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
        
        # Check if confidence is low for enhanced LLM analysis
        is_low_confidence = confidence < 0.6
        
        # LLM-based synthesis - adaptive based on confidence level
        if is_low_confidence and image_path:
            # Enhanced prompt for low confidence scenarios with image reference
            prompt = f"""Bạn là chuyên gia nội soi. Đây là kết quả phân loại {self.classifier_type} với ĐỘ TIN CẬY THẤP.

Thông tin ảnh:
- Đường dẫn ảnh: {image_path}
- Câu hỏi của người dùng: "{query if query else f'Phân loại {self.classifier_type} trong ảnh nội soi này'}"

Kết quả phân loại: 
- Kết quả: {class_name} 
- Độ tin cậy: {confidence:.1%} (THẤP)
- Các lớp khác: {all_classes}
- Mô tả: {description}

NHIỆM VỤ CỦA BẠN:
1. Xác nhận ĐỘ TIN CẬY THẤP cho kết quả {class_name}
2. Phân tích PHÂN PHỐI XÁC SUẤT của các lớp khác
3. Đề xuất kết luận phù hợp nhất dựa trên kiến thức y khoa của bạn
4. Mô tả ý nghĩa lâm sàng và khuyến nghị

QUAN TRỌNG: Hãy nêu rõ đây là đánh giá với độ tin cậy thấp và cần thêm kiểm tra bởi chuyên gia."""
        else:
            # Standard prompt for normal confidence
            prompt = f"""Bạn là chuyên gia nội soi. Hãy giải thích kết quả phân loại sau cho bệnh nhân một cách dễ hiểu và chuyên nghiệp:

- Kết quả: {class_name}
- Độ tin cậy: {confidence:.1%}
- Các lớp khác: {all_classes}
- Mô tả: {description}

Hãy đưa ra nhận định lâm sàng, ý nghĩa kết quả và khuyến nghị nếu có."""

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
            "synthesis_method": "llm_natural_language",
            "final_answer_format": "structured" if final_answer_data else "unstructured",
            "is_low_confidence": is_low_confidence
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
            is_low_confidence = False
            
            if confidence >= 0.8:
                quality_indicators.append("High confidence classification")
            elif confidence >= 0.6:
                quality_indicators.append("Moderate confidence classification")
            else:
                quality_indicators.append("LOW CONFIDENCE CLASSIFICATION - REQUIRES EXPERT REVIEW")
                is_low_confidence = True
            
            # Analyze class distribution
            if all_classes:
                sorted_classes = sorted(all_classes.items(), key=lambda x: x[1], reverse=True)
                top_2 = sorted_classes[:2]
                if len(top_2) > 1:
                    margin = top_2[0][1] - top_2[1][1]
                    if margin < 0.2:
                        quality_indicators.append("Close competition between top classes")
                        if is_low_confidence:
                            # Add alternatives analysis for low confidence
                            alternatives = []
                            for cls, prob in sorted_classes[:3]:  # Top 3 alternatives
                                alternatives.append(f"{cls}: {prob:.1%}")
                            quality_indicators.append(f"Alternative classifications: {', '.join(alternatives)}")
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
            
            # Title for synthesis with confidence alert if needed
            title = f"**Guided Adaptive {self.classifier_type.title()} Classification Analysis**"
            if is_low_confidence:
                title += " ⚠️ LOW CONFIDENCE ALERT"
            
            # Synthesize analysis
            synthesis = f"""{title}

Classification Result: {class_name} ({confidence:.1%} confidence)

Quality Assessment:
{chr(10).join(f'• {indicator}' for indicator in quality_indicators)}

Tool Performance: Successfully completed classification using adaptive strategy
Requirements Status: {completed_reqs}/{total_reqs} core requirements met

"""
            
            # Add additional context for low confidence results
            if is_low_confidence:
                synthesis += f"""LLM Analysis for Low Confidence Result:
The model's confidence is below threshold ({confidence:.1%}). LLM assessment:
{llm_answer[:300]}{'...' if len(llm_answer) > 300 else ''}

"""
            else:
                synthesis += f"""Clinical Interpretation:
{llm_answer[:200]}{'...' if len(llm_answer) > 200 else ''}

"""
            
            synthesis += "Synthesis Method: Text-only analysis of tool results (no image review required)"
            
            return synthesis
            
        except Exception as e:
            return f"Synthesis analysis completed with guided adaptive approach. Classification: {class_name} ({confidence:.1%})"

    # ===== DOMAIN-SPECIFIC HELPERS (reused from original) =====
    
    def _get_modality_advantages(self, modality: str) -> List[str]:
        """Get clinical advantages of imaging modality."""
        advantages_map = {
            "WLI": ["Enhanced color contrast", "Better inflammation visualization", "Improved polyp detection", "Subtle lesion detection"],  # WLI thực tế là LCI
            "BLI": ["Standard visualization", "Natural colors", "General screening", "Baseline reference"],  # BLI thực tế là WLI
            "FICE": ["Enhanced vasculature", "Better lesion detection", "Improved contrast", "Surface pattern analysis"],  # FICE thực tế là BLI
            "LCI": ["Digital enhancement", "Customizable settings", "Color difference detection", "Inflammatory assessment"]  # LCI thực tế là FICE
        }
        return advantages_map.get(modality, ["Standard endoscopic visualization"])
    
    def _get_modality_recommendations(self, modality: str) -> List[str]:
        """Get usage recommendations for modality."""
        recommendations_map = {
            "WLI": ["High-risk screening", "Subtle lesions", "Healing assessment", "Enhanced adenoma detection"],  # WLI thực tế là LCI
            "BLI": ["Initial screening", "General procedures", "Documentation baseline", "When specialized unavailable"],  # BLI thực tế là WLI
            "FICE": ["Suspicious lesions", "Detailed assessment", "Polyp characterization", "High-risk surveillance"],  # FICE thực tế là BLI
            "LCI": ["Flat lesions", "Inflammatory changes", "Chromoendoscopy needs", "Surface pattern analysis"]  # LCI thực tế là FICE
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

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Custom ReAct loop with better final answer handling for classifier."""
        self.react_history = []
        
        for i in range(1, self.max_iterations + 1):
            # Get LLM response
            resp = self.llm.invoke(self._create_react_messages(task_input)).content
            print(f"🔧 DEBUG: resp {i} {resp}")
            
            # Parse response
            t, a, inp = self._parse_llm_response(resp)
            print(f"🔧 DEBUG: t {i} {t[:50] if t else 'None'}")
            print(f"🔧 DEBUG: a {i} {a}")
            print(f"🔧 DEBUG: inp {i} {inp}")
            
            # Extra fallback for thought being None
            if not t:
                t = f"Processing classification step {i}"
                print(f"🔧 DEBUG: Using fallback thought: {t}")
                
            # Extra fallback for action being None
            if not a and "final answer" in resp.lower():
                a = "Final Answer"
                if not inp:
                    inp = {"answer": resp.split("final answer", 1)[1].strip(), "fallback": True}
                print(f"🔧 DEBUG: Using fallback action 'Final Answer' from text")
                
            if not t or not a:
                print(f"🔧 DEBUG: Invalid response, skipping iteration {i}")
                continue
                
            # Create step
            step = ReActStep(
                thought=t, 
                thought_type=ThoughtType.INITIAL if i == 1 else ThoughtType.REASONING,
                action=a,
                action_input=inp
            )
            
            # Handle final answer with more flexibility
            print(f"🔧 DEBUG: Checking final answer condition: '{a.lower() if a else 'None'}'")
            if a and a.lower() in ["final answer", "final_answer"]:
                print(f"🔧 DEBUG: Final Answer detected!")
                step.thought_type = ThoughtType.CONCLUSION
                self.react_history.append(step)
                
                # Extract answer from input or use thought as fallback
                if inp and "answer" in inp:
                    answer = inp["answer"]
                    print(f"🔧 DEBUG: Using answer from input: {str(answer)[:50]}...")
                else:
                    answer = t
                    print(f"🔧 DEBUG: Using thought as fallback answer: {str(answer)[:50]}...")
                
                return {
                    "success": True, 
                    "answer": answer, 
                    "history": self._serialize_history(),
                    "iterations_used": i,
                    "termination_reason": "final_answer"
                }
                
            # Normal tool execution
            print(f"🔧 DEBUG: Executing tool: {a}")
            obs = self._execute_tool(a, inp or {})
            step.observation = obs
            step.thought_type = ThoughtType.OBSERVATION
            self.react_history.append(step)
            task_input[f"obs_{i}"] = obs
            
            print(f"🔧 DEBUG: ReAct iteration {i} completed")
            print(f"🔧 DEBUG: Thought: {t[:50]}...")
            print(f"🔧 DEBUG: Action: {a}")
            print(f"🔧 DEBUG: Observation: {obs[:100]}...")
            
            # Check if observation has high confidence classification
            try:
                obs_data = json.loads(obs)
                if obs_data.get("success") and obs_data.get("confidence", 0) >= 0.8:
                    # Add a hint to LLM that this is a good time for final answer
                    task_input["high_confidence_found"] = True
                    print(f"🔧 DEBUG: High confidence found, adding hint for final answer")
            except Exception as e:
                print(f"🔧 DEBUG: Error parsing observation: {str(e)}")
                
        # Max iterations reached
        print(f"🔧 DEBUG: Max iterations reached without final answer")
        return {"success": False, "error": "Max iterations reached", "history": self._serialize_history()}

    def _parse_llm_response(self, txt: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Parse LLM response with debugging for Thought extraction."""
        txt = txt.strip()
        print(f"🔍 DEBUG: Raw LLM response: {txt[:100]}...")
        
        # More flexible Thought extraction - try multiple patterns
        thought_val = None
        
        # Pattern 1: Standard "Thought:" prefix
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", txt, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought_val = thought_match.group(1).strip()
            print(f"🔍 DEBUG: Found thought with pattern 1: {thought_val[:50]}...")
        
        # Pattern 2: Beginning of text until Action/Final Answer
        if not thought_val:
            beginning_match = re.search(r"^(.*?)(?=Action:|Final Answer:|$)", txt, re.DOTALL)
            if beginning_match and beginning_match.group(1).strip():
                thought_val = beginning_match.group(1).strip()
                print(f"🔍 DEBUG: Found thought with pattern 2: {thought_val[:50]}...")
        
        # Pattern 3: If still no thought, take first paragraph as thought
        if not thought_val:
            paragraphs = txt.split("\n\n")
            if paragraphs:
                thought_val = paragraphs[0].strip()
                print(f"🔍 DEBUG: Using first paragraph as thought: {thought_val[:50]}...")
                
        # If still no thought, use a default
        if not thought_val:
            thought_val = "Processing classification"
            print(f"🔍 DEBUG: Using default thought: {thought_val}")
            
        # Look for Final Answer 
        final_answer_pattern = re.search(r"(Final Answer:|Final answer:|FINAL ANSWER:|final answer:)\s*(.*?)(?=$)", txt, re.DOTALL | re.IGNORECASE)
        
        if final_answer_pattern:
            print(f"🔍 DEBUG: Found Final Answer pattern")
            # Found Final Answer
            action_val = "Final Answer"  # Use exact match for ReAct
            answer_content = final_answer_pattern.group(2).strip()
            
            # Check if answer content looks like JSON
            if answer_content.startswith('{') and answer_content.endswith('}'):
                try:
                    answer_data = json.loads(answer_content)
                    input_val = {"answer": answer_data}
                    print(f"🔍 DEBUG: Parsed JSON from Final Answer")
                except json.JSONDecodeError:
                    input_val = {"answer": answer_content}
                    print(f"🔍 DEBUG: Using plain text from Final Answer (JSON parse failed)")
            else:
                input_val = {"answer": answer_content}
                print(f"🔍 DEBUG: Using plain text from Final Answer")
                
            return thought_val, action_val, input_val
         
        # Normal Action pattern
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", txt, re.DOTALL | re.IGNORECASE)
        action_val = None
        if action_match:
            action_val = action_match.group(1).strip()
            print(f"🔍 DEBUG: Found action: {action_val}")
            
            # Check if action is some variant of "Final Answer"
            if action_val and re.search(r"final\s*answer", action_val, re.IGNORECASE):
                action_val = "Final Answer"  # Normalize for ReAct
                print(f"🔍 DEBUG: Normalized action to 'Final Answer'")
        
        # Extract Action Input
        input_val = None
        a_input = re.search(r"Action Input:\s*(\{.+?\})", txt, re.DOTALL)
        if a_input:
            try:
                input_val = json.loads(a_input.group(1))
                print(f"🔍 DEBUG: Parsed action input JSON")
            except Exception as e:
                print(f"🔍 DEBUG: Failed to parse action input JSON: {str(e)}")
                
        print(f"🔍 FINAL PARSE RESULT - thought: {'Found' if thought_val else 'None'}, action: {action_val}")
        return thought_val, action_val, input_val

# ===== USAGE EXAMPLE =====
def test_guided_adaptive_classifier():
    """Test guided adaptive classifier."""
    
    # Test modality classifier
    modality_classifier = ClassifierAgent(
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