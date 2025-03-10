"""
Plan-related classes for the agent loop.

This module provides classes for creating and managing plans for completing tasks.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A step in the plan."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])  # Unique identifier for the step
    description: str
    completed: bool = False
    parent_id: Optional[str] = None  # ID of parent step if this is a substep
    substeps: List["PlanStep"] = Field(default_factory=list)
    completion_time: Optional[datetime] = None  # When the step was completed


PlanStep.update_forward_refs()


class Plan(BaseModel):
    """A plan for completing a task."""

    task: str
    steps: List[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    current_step_id: Optional[str] = None  # Track the current step being worked on
    completed_steps: List[str] = Field(default_factory=list)  # List of completed step IDs
    
    def get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """Find a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
            if step.substeps:
                for substep in step.substeps:
                    if substep.id == step_id:
                        return substep
        return None
    
    def get_next_incomplete_step(self) -> Optional[PlanStep]:
        """Get the next incomplete step in the plan."""
        for step in self.steps:
            if not step.completed:
                return step
            if step.substeps:
                for substep in step.substeps:
                    if not substep.completed:
                        return substep
        return None
    
    def mark_step_complete(self, step_id: str) -> bool:
        """Mark a step as complete by its ID."""
        step = self.get_step_by_id(step_id)
        if step:
            step.completed = True
            step.completion_time = datetime.now()
            self.completed_steps.append(step_id)
            self.last_updated = datetime.now()
            return True
        return False
    
    def to_markdown(self) -> str:
        """Convert the plan to markdown format."""
        lines = [f"# Plan for: {self.task}\n", "## Steps:"]
        
        for step in self.steps:
            checkbox = "[x]" if step.completed else "[ ]"
            lines.append(f"- {checkbox} {step.description}")
            
            if step.substeps:
                for substep in step.substeps:
                    checkbox = "[x]" if substep.completed else "[ ]"
                    lines.append(f"  - {checkbox} {substep.description}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_markdown(cls, markdown: str) -> "Plan":
        """
        Parse a markdown plan into a structured Plan object.
        
        Args:
            markdown: Markdown string containing the plan
            
        Returns:
            Plan object
        """
        lines = markdown.strip().split("\n")
        
        # Extract task from the title
        task = ""
        for line in lines:
            if line.startswith("# Plan for:"):
                task = line.replace("# Plan for:", "").strip()
                break
        
        plan = cls(task=task)
        
        # Parse steps and substeps
        current_step = None
        for line in lines:
            # Skip empty lines and headers
            if not line or line.startswith("#"):
                continue
                
            # Check if it's a step or substep
            if line.startswith("- "):
                # It's a main step
                step_text = line[2:].strip()
                completed = False
                
                # Check if it's completed
                if step_text.startswith("[x]"):
                    completed = True
                    step_text = step_text[3:].strip()
                elif step_text.startswith("[ ]"):
                    step_text = step_text[3:].strip()
                
                # Create the step
                current_step = PlanStep(
                    description=step_text,
                    completed=completed
                )
                
                # Add to the plan
                plan.steps.append(current_step)
                
                # If completed, add to completed steps
                if completed:
                    plan.completed_steps.append(current_step.id)
                
            elif line.lstrip().startswith("- ") and current_step and line.startswith("  "):
                # It's a substep (indented with spaces)
                substep_text = line.lstrip()[2:].strip()
                completed = False
                
                # Check if it's completed
                if substep_text.startswith("[x]"):
                    completed = True
                    substep_text = substep_text[3:].strip()
                elif substep_text.startswith("[ ]"):
                    substep_text = substep_text[3:].strip()
                
                # Create the substep
                substep = PlanStep(
                    description=substep_text,
                    completed=completed,
                    parent_id=current_step.id
                )
                
                # Add to the current step
                current_step.substeps.append(substep)
                
                # If completed, add to completed steps
                if completed:
                    plan.completed_steps.append(substep.id)
        
        # Set the current step to the first incomplete step
        next_step = plan.get_next_incomplete_step()
        if next_step:
            plan.current_step_id = next_step.id
            
        return plan
    
    def find_step_by_description(self, description: str) -> Optional[PlanStep]:
        """
        Find a step by its description (or partial match).
        
        Args:
            description: Description to match
            
        Returns:
            Matching PlanStep or None
        """
        # First try exact match
        for step in self.steps:
            if step.description.lower() == description.lower():
                return step
            
            for substep in step.substeps:
                if substep.description.lower() == description.lower():
                    return substep
        
        # Then try partial match
        best_match = None
        best_score = 0
        
        for step in self.steps:
            score = self._similarity_score(step.description.lower(), description.lower())
            if score > best_score:
                best_match = step
                best_score = score
            
            for substep in step.substeps:
                score = self._similarity_score(substep.description.lower(), description.lower())
                if score > best_score:
                    best_match = substep
                    best_score = score
        
        # Only return if we have a reasonable match
        if best_score > 0.5:
            return best_match
            
        return None
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two strings.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check if one is a substring of the other
        if text1 in text2 or text2 in text1:
            return 0.8
            
        # Count matching words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2)) 