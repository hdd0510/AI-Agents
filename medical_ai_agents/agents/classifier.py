#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Classifier Agent with ReAct Pattern
------------------------------------------------------
Agent phân loại hình ảnh nội soi sử dụng ReAct framework theo BaseAgent mới.
"""

import json
from typing import Dict, Any, List
import logging

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.classifier.cls_tools import ClassifierTool


class ClassifierAgent(BaseAgent):
    """Classifier Agent với ReAct pattern cho phân loại hình ảnh nội soi."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", llm_model: str = "gpt-4o-mini", 
                device: str = "cuda"):
        """Initialize Classifier ReAct Agent."""
        self.model_path = model_path
        self.class_names = class_names
        self.classifier_type = classifier_type
        
        # Set specific agent name based on type
        agent_name = f"{classifier_type.capitalize()} Classifier Agent"
        super().__init__(name=agent_name, llm_model=llm_model, device=device)
        
        # Specific configuration
        self.max_iterations = 3  # Classification typically needs fewer steps
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
        """Get classifier agent description."""
        if self.classifier_type == "modality":
            return """I am a medical imaging modality classification specialist.
My expertise includes:
- Identifying endoscopy imaging techniques (WLI, BLI, FICE, LCI)
- Understanding the characteristics of each imaging modality
- Explaining the clinical advantages of different techniques
- Recommending appropriate modality for specific diagnostic needs
- Assessing image quality and technical parameters"""
        else:  # region classifier
            return """I am an anatomical region classification specialist for gastrointestinal endoscopy.
My expertise includes:
- Identifying anatomical locations in the GI tract
- Understanding anatomical landmarks and transitions
- Explaining the clinical significance of different regions
- Correlating anatomical location with pathology risk
- Providing location-specific screening recommendations"""
    
    def initialize(self) -> bool:
        """Initialize classifier agent."""
        try:
            self.initialized = True
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {str(e)}")
            self.initialized = False
            return False
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract classification task input."""
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "classifier_type": self.classifier_type,
            "previous_results": self._get_previous_results(state)
        }
    
    def _get_previous_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get results from previous agents for context."""
        previous = {}
        
        # Get detector results if available
        if "detector_result" in state:
            detector = state["detector_result"]
            if detector.get("success", False):
                previous["polyp_count"] = detector.get("count", 0)
                if detector.get("objects"):
                    previous["polyp_locations"] = [
                        obj.get("position_description", "unknown") 
                        for obj in detector["objects"][:3]
                    ]
        
        # Get other classifier result if this is the second classifier
        if self.classifier_type == "region" and "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                previous["imaging_modality"] = modality.get("class_name", "Unknown")
        elif self.classifier_type == "modality" and "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                previous["anatomical_region"] = region.get("class_name", "Unknown")
        
        return previous
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input cho ReAct processing (abstract method implementation)."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        classifier_type = task_input.get("classifier_type", "")
        previous = task_input.get("previous_results", {})
        
        # Build context from previous results
        context_parts = []
        if previous:
            if "polyp_count" in previous:
                context_parts.append(f"- Polyp detection: {previous['polyp_count']} polyp(s) found")
            if "polyp_locations" in previous:
                context_parts.append(f"- Polyp locations: {', '.join(previous['polyp_locations'])}")
            if "imaging_modality" in previous:
                context_parts.append(f"- Imaging technique: {previous['imaging_modality']}")
            if "anatomical_region" in previous:
                context_parts.append(f"- Anatomical location: {previous['anatomical_region']}")
        
        context_str = "\n".join(context_parts) if context_parts else "No previous analysis results"
        
        # Build task description based on classifier type
        if classifier_type == "modality":
            task_desc = """Classify the endoscopy imaging modality/technique.
Possible modalities:
- WLI (White Light Imaging): Standard white light endoscopy
- BLI (Blue Light Imaging): Enhanced visualization of blood vessels and surface patterns
- FICE (Flexible spectral Imaging Color Enhancement): Digital chromoendoscopy
- LCI (Linked Color Imaging): Enhanced color contrast for improved lesion detection"""
        else:  # region
            task_desc = """Classify the anatomical region in the gastrointestinal tract.
Possible regions:
- Hau_hong (Pharynx): Throat region
- Thuc_quan (Esophagus): Tube connecting throat to stomach
- Tam_vi (Cardia): Junction between esophagus and stomach
- Than_vi (Body): Main part of stomach
- Phinh_vi (Fundus): Upper curved portion of stomach
- Hang_vi (Antrum): Lower portion of stomach
- Bo_cong_lon/Bo_cong_nho: Greater/Lesser curvature of stomach
- Hanh_ta_trang (Duodenal bulb): First part of duodenum
- Ta_trang (Duodenum): First section of small intestine"""
        
        prompt = f"""**Medical Image Classification Task**

Image to classify: {image_path}
Classification type: {classifier_type}
User query: "{query if query else f'Please classify the {classifier_type} of this endoscopy image'}"

Previous analysis results:
{context_str}

{task_desc}

Requirements:
1. Use the {classifier_type}_classifier tool to classify the image
2. Explain the visual characteristics that led to the classification
3. Discuss clinical significance of the identified {classifier_type}
4. Consider how the classification relates to any previous findings

Please proceed with the classification analysis using ReAct pattern."""
        
        return prompt
    
    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format ReAct result into classifier agent output."""
        result_key = f"{self.classifier_type}_result"
        
        if not react_result.get("success", False):
            return {
                result_key: {
                    "success": False,
                    "error": react_result.get("error", "Classification failed"),
                    "reasoning_steps": len(self.react_history) if hasattr(self, 'react_history') else 0
                }
            }
        
        # Extract classification data from ReAct history
        class_name = "Unknown"
        confidence = 0.0
        all_classes = {}
        description = ""
        
        for step in self.react_history:
            if step.observation:
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
        
        # Get the comprehensive analysis from final answer
        final_analysis = react_result.get("answer", "")
        
        # Build structured result
        classifier_result = {
            "success": True,
            "class_name": class_name,
            "confidence": confidence,
            "description": description,
            "all_classes": all_classes,
            "analysis": final_analysis,
            "reasoning_steps": len(self.react_history) if hasattr(self, 'react_history') else 0
        }
        
        # Add type-specific insights
        if self.classifier_type == "modality":
            classifier_result["clinical_advantages"] = self._get_modality_advantages(class_name)
            classifier_result["recommended_for"] = self._get_modality_recommendations(class_name)
        else:  # region
            classifier_result["anatomical_significance"] = self._get_region_significance(class_name)
            classifier_result["pathology_risk"] = self._get_region_risk(class_name)
        
        return {result_key: classifier_result}
    
    # ===== DOMAIN-SPECIFIC HELPERS (từ version cũ) =====
    
    def _get_modality_advantages(self, modality: str) -> List[str]:
        """Get clinical advantages of the imaging modality."""
        advantages = {
            "WLI": [
                "Standard visualization with natural colors",
                "Good for general screening",
                "No special equipment required",
                "Baseline reference for comparison"
            ],
            "BLI": [
                "Enhanced visualization of microvasculature",
                "Better detection of early neoplastic lesions",
                "Improved contrast for mucosal patterns",
                "Superior for characterizing surface irregularities"
            ],
            "FICE": [
                "Digital enhancement without dyes",
                "Customizable spectral settings",
                "Good for detecting subtle color differences",
                "Effective for inflammatory changes"
            ],
            "LCI": [
                "Enhanced color contrast",
                "Better visualization of inflammation",
                "Improved polyp detection rates",
                "Excellent for subtle lesion detection"
            ]
        }
        return advantages.get(modality, ["Standard endoscopic visualization"])
    
    def _get_modality_recommendations(self, modality: str) -> List[str]:
        """Get recommendations for when to use this modality."""
        recommendations = {
            "WLI": [
                "Initial screening examination",
                "General diagnostic procedures",
                "When specialized equipment unavailable",
                "Documentation and comparison baseline"
            ],
            "BLI": [
                "Evaluation of suspicious lesions",
                "Detailed mucosal assessment",
                "Characterization of polyps",
                "Surveillance of high-risk patients"
            ],
            "FICE": [
                "Detection of flat lesions",
                "Assessment of inflammatory changes",
                "When chromoendoscopy is needed",
                "Detailed surface pattern analysis"
            ],
            "LCI": [
                "Screening in high-risk patients",
                "Detection of subtle lesions",
                "Evaluation of healing mucosa",
                "Enhanced adenoma detection"
            ]
        }
        return recommendations.get(modality, ["Consult with endoscopist for optimal technique"])
    
    def _get_region_significance(self, region: str) -> str:
        """Get anatomical significance of the region."""
        significance = {
            "Hau_hong": "Entry point of digestive system, important for swallowing function and respiratory interface",
            "Thuc_quan": "Critical conduit for food transport, common site for reflux disease and Barrett's esophagus",
            "Tam_vi": "Gastroesophageal junction, prone to inflammation and adenocarcinoma development",
            "Than_vi": "Main gastric body for acid production and digestion, common site for ulcers and tumors",
            "Phinh_vi": "Gastric fundus for storage, can develop fundic gland polyps and varices",
            "Hang_vi": "Gastric antrum, high-risk area for Helicobacter pylori and gastric cancer",
            "Bo_cong_lon": "Greater curvature, less common site for pathology but important for surgical planning",
            "Bo_cong_nho": "Lesser curvature, high-risk area for gastric cancer and lymph node involvement",
            "Hanh_ta_trang": "Duodenal bulb, common site for duodenal ulcers and Brunner's gland hyperplasia",
            "Ta_trang": "Duodenum proper, important for nutrient absorption and celiac disease manifestation"
        }
        return significance.get(region, "Important anatomical region of the gastrointestinal tract")
    
    def _get_region_risk(self, region: str) -> Dict[str, str]:
        """Get pathology risk assessment for the region."""
        risk_profiles = {
            "Hau_hong": {
                "cancer_risk": "Low", 
                "common_pathology": "Pharyngitis, foreign bodies, reflux changes"
            },
            "Thuc_quan": {
                "cancer_risk": "Moderate", 
                "common_pathology": "GERD, Barrett's esophagus, squamous cell carcinoma, adenocarcinoma"
            },
            "Tam_vi": {
                "cancer_risk": "Moderate-High", 
                "common_pathology": "Carditis, intestinal metaplasia, adenocarcinoma"
            },
            "Than_vi": {
                "cancer_risk": "Moderate", 
                "common_pathology": "Gastritis, peptic ulcers, MALT lymphoma, adenocarcinoma"
            },
            "Phinh_vi": {
                "cancer_risk": "Low", 
                "common_pathology": "Fundic gland polyps, portal hypertensive gastropathy, varices"
            },
            "Hang_vi": {
                "cancer_risk": "High", 
                "common_pathology": "H. pylori gastritis, intestinal metaplasia, adenoma, adenocarcinoma"
            },
            "Bo_cong_lon": {
                "cancer_risk": "Low-Moderate", 
                "common_pathology": "Gastric ulcers, GIST, lymphoma"
            },
            "Bo_cong_nho": {
                "cancer_risk": "High", 
                "common_pathology": "Gastric ulcers, adenocarcinoma, lymph node metastases"
            },
            "Hanh_ta_trang": {
                "cancer_risk": "Low", 
                "common_pathology": "Duodenal ulcers, Brunner's gland hyperplasia, duodenitis"
            },
            "Ta_trang": {
                "cancer_risk": "Low", 
                "common_pathology": "Duodenitis, celiac disease, adenocarcinoma (rare)"
            }
        }
        return risk_profiles.get(region, {
            "cancer_risk": "Variable", 
            "common_pathology": "Various gastrointestinal conditions"
        })