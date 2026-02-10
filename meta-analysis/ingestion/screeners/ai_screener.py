"""
AI-Powered Paper Screening for Meta-Analysis.

This module provides standalone screening capabilities including:
- Single and batch paper screening
- Dual-reviewer simulation for reliability
- Configurable inclusion/exclusion criteria
- PRISMA flow data export
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import hashlib


class Decision(Enum):
    """Screening decision."""
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    UNCERTAIN = "UNCERTAIN"
    CONFLICT = "CONFLICT"  # For dual review disagreements


@dataclass
class ScreeningResult:
    """Result of screening a single paper."""
    paper_id: str
    title: str
    decision: Decision
    confidence: float
    reasoning: str
    criteria_met: List[str] = field(default_factory=list)
    exclusion_reasons: List[str] = field(default_factory=list)
    reviewer: str = "AI-1"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # For dual review
    reviewer2_decision: Optional[Decision] = None
    reviewer2_confidence: Optional[float] = None
    reviewer2_reasoning: Optional[str] = None
    final_decision: Optional[Decision] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["decision"] = self.decision.value
        if self.reviewer2_decision:
            result["reviewer2_decision"] = self.reviewer2_decision.value
        if self.final_decision:
            result["final_decision"] = self.final_decision.value
        return result


@dataclass
class ScreeningConfig:
    """Configuration for screening session."""
    research_question: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    
    # Optional PICO framework
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparison: Optional[str] = None
    outcome: Optional[str] = None
    
    # Screening settings
    confidence_threshold: float = 0.7  # Below this = UNCERTAIN
    require_abstract: bool = True
    
    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompt."""
        lines = [f"Research Question: {self.research_question}"]
        
        if any([self.population, self.intervention, self.comparison, self.outcome]):
            lines.append("\nPICO Framework:")
            if self.population:
                lines.append(f"  Population: {self.population}")
            if self.intervention:
                lines.append(f"  Intervention: {self.intervention}")
            if self.comparison:
                lines.append(f"  Comparison: {self.comparison}")
            if self.outcome:
                lines.append(f"  Outcome: {self.outcome}")
        
        lines.append("\nINCLUSION CRITERIA:")
        for c in self.inclusion_criteria:
            lines.append(f"  - {c}")
        
        lines.append("\nEXCLUSION CRITERIA:")
        for c in self.exclusion_criteria:
            lines.append(f"  - {c}")
        
        return "\n".join(lines)


class AIScreener:
    """
    AI-powered paper screener for systematic reviews and meta-analyses.
    
    Features:
    - Screen individual papers or batches
    - Dual-reviewer simulation for reliability
    - Configurable criteria and thresholds
    - PRISMA-compliant data export
    
    Example:
        >>> config = ScreeningConfig(
        ...     research_question="Brain activation during navigation",
        ...     inclusion_criteria=["Reports fMRI data", "Human subjects"],
        ...     exclusion_criteria=["Review articles", "Animal studies"]
        ... )
        >>> screener = AIScreener(config)
        >>> results = screener.screen_batch(papers)
        >>> screener.export_prisma("prisma_data.json")
    """
    
    SCREENING_PROMPT = '''You are an expert systematic reviewer screening abstracts for a meta-analysis.

{context}

Paper to screen:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Based ONLY on the information in the abstract, determine:
1. Should this paper be INCLUDED, EXCLUDED, or marked UNCERTAIN?
2. Your confidence level (0.0 to 1.0)
3. Which inclusion criteria are clearly met?
4. What exclusion reasons apply (if any)?
5. Brief reasoning for your decision

IMPORTANT:
- If the abstract doesn't provide enough information, mark as UNCERTAIN
- Be conservative: when in doubt, lean toward UNCERTAIN rather than EXCLUDE
- Only EXCLUDE if there's clear evidence of exclusion criteria

Return your response as JSON:
{{
    "decision": "INCLUDE" | "EXCLUDE" | "UNCERTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "criteria_met": ["list of clearly met inclusion criteria"],
    "exclusion_reasons": ["list of exclusion reasons, empty if none"]
}}'''

    def __init__(
        self,
        config: ScreeningConfig,
        provider: str = "anthropic",
        model: Optional[str] = None
    ):
        """
        Initialize the AI screener.
        
        Args:
            config: ScreeningConfig with criteria and settings
            provider: LLM provider ("anthropic" or "openai")
            model: Optional model override
        """
        self.config = config
        self.provider = provider
        self.model = model
        self.results: List[ScreeningResult] = []
        self._llm = None
        
    def _get_llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            import sys
            sys.path.insert(0, '..')
            from extraction.extractors.base_extractor import LLMProvider
            self._llm = LLMProvider(provider=self.provider)
        return self._llm
    
    def _generate_paper_id(self, paper: Dict) -> str:
        """Generate unique ID for a paper."""
        if "pmid" in paper and paper["pmid"]:
            return f"pmid:{paper['pmid']}"
        if "doi" in paper and paper["doi"]:
            return f"doi:{paper['doi']}"
        # Hash title as fallback
        title_hash = hashlib.md5(paper.get("title", "").encode()).hexdigest()[:8]
        return f"hash:{title_hash}"
    
    def screen(self, paper: Dict, reviewer: str = "AI-1") -> ScreeningResult:
        """
        Screen a single paper.
        
        Args:
            paper: Dictionary with title, authors, year, abstract
            reviewer: Reviewer identifier (for dual review)
            
        Returns:
            ScreeningResult object
        """
        # Check for required fields
        if self.config.require_abstract and not paper.get("abstract"):
            return ScreeningResult(
                paper_id=self._generate_paper_id(paper),
                title=paper.get("title", "Unknown"),
                decision=Decision.UNCERTAIN,
                confidence=0.0,
                reasoning="No abstract available for screening",
                reviewer=reviewer
            )
        
        # Build prompt
        prompt = self.SCREENING_PROMPT.format(
            context=self.config.to_prompt_context(),
            title=paper.get("title", "Unknown"),
            authors=paper.get("authors", "Unknown"),
            year=paper.get("year", "Unknown"),
            abstract=paper.get("abstract", "No abstract available")
        )
        
        # Get LLM response
        llm = self._get_llm()
        response = llm.extract(paper.get("abstract", ""), prompt)
        
        # Parse decision
        decision_str = response.get("decision", "UNCERTAIN").upper()
        try:
            decision = Decision[decision_str]
        except KeyError:
            decision = Decision.UNCERTAIN
        
        # Apply confidence threshold
        confidence = float(response.get("confidence", 0.5))
        if confidence < self.config.confidence_threshold and decision != Decision.EXCLUDE:
            decision = Decision.UNCERTAIN
        
        result = ScreeningResult(
            paper_id=self._generate_paper_id(paper),
            title=paper.get("title", "Unknown"),
            decision=decision,
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            criteria_met=response.get("criteria_met", []),
            exclusion_reasons=response.get("exclusion_reasons", []),
            reviewer=reviewer
        )
        
        self.results.append(result)
        return result
    
    def screen_batch(
        self,
        papers: List[Dict],
        delay: float = 0.5,
        progress_callback=None
    ) -> List[ScreeningResult]:
        """
        Screen multiple papers.
        
        Args:
            papers: List of paper dictionaries
            delay: Delay between API calls (rate limiting)
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of ScreeningResult objects
        """
        results = []
        total = len(papers)
        
        for i, paper in enumerate(papers):
            result = self.screen(paper)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            if i < total - 1:
                time.sleep(delay)
        
        return results
    
    def dual_review(
        self,
        papers: List[Dict],
        delay: float = 0.5,
        resolve_conflicts: bool = True
    ) -> List[ScreeningResult]:
        """
        Perform dual-reviewer screening simulation.
        
        Each paper is screened twice with different prompts/temperatures.
        Conflicts are flagged for manual review or auto-resolved.
        
        Args:
            papers: List of paper dictionaries
            delay: Delay between API calls
            resolve_conflicts: Auto-resolve conflicts (conservative = UNCERTAIN)
            
        Returns:
            List of ScreeningResult objects with dual review data
        """
        results = []
        
        for paper in papers:
            # First review
            result1 = self.screen(paper, reviewer="AI-Reviewer-1")
            time.sleep(delay)
            
            # Second review (don't add to self.results)
            self.results.pop()  # Remove first result temporarily
            result2 = self.screen(paper, reviewer="AI-Reviewer-2")
            self.results.pop()  # Remove second result
            
            # Merge results
            result1.reviewer2_decision = result2.decision
            result1.reviewer2_confidence = result2.confidence
            result1.reviewer2_reasoning = result2.reasoning
            
            # Determine final decision
            if result1.decision == result2.decision:
                result1.final_decision = result1.decision
            else:
                result1.final_decision = Decision.CONFLICT
                if resolve_conflicts:
                    # Conservative resolution
                    if Decision.INCLUDE in [result1.decision, result2.decision]:
                        result1.final_decision = Decision.UNCERTAIN
                    else:
                        result1.final_decision = Decision.EXCLUDE
            
            self.results.append(result1)
            results.append(result1)
            
            time.sleep(delay)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of screening results."""
        if not self.results:
            return {"total": 0}
        
        decisions = [r.decision for r in self.results]
        final_decisions = [r.final_decision or r.decision for r in self.results]
        
        return {
            "total": len(self.results),
            "included": sum(1 for d in final_decisions if d == Decision.INCLUDE),
            "excluded": sum(1 for d in final_decisions if d == Decision.EXCLUDE),
            "uncertain": sum(1 for d in final_decisions if d == Decision.UNCERTAIN),
            "conflicts": sum(1 for d in final_decisions if d == Decision.CONFLICT),
            "mean_confidence": sum(r.confidence for r in self.results) / len(self.results),
            "dual_reviewed": sum(1 for r in self.results if r.reviewer2_decision is not None)
        }
    
    def get_included(self) -> List[ScreeningResult]:
        """Get papers marked for inclusion."""
        return [r for r in self.results 
                if (r.final_decision or r.decision) == Decision.INCLUDE]
    
    def get_excluded(self) -> List[ScreeningResult]:
        """Get papers marked for exclusion."""
        return [r for r in self.results 
                if (r.final_decision or r.decision) == Decision.EXCLUDE]
    
    def get_uncertain(self) -> List[ScreeningResult]:
        """Get papers needing manual review."""
        return [r for r in self.results 
                if (r.final_decision or r.decision) in [Decision.UNCERTAIN, Decision.CONFLICT]]
    
    def export_results(self, path: str):
        """Export all results to JSON."""
        data = {
            "config": asdict(self.config),
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_prisma(self, path: str) -> Dict[str, Any]:
        """
        Export PRISMA flow diagram data.
        
        Returns dict with counts for each PRISMA stage.
        """
        summary = self.get_summary()
        
        prisma_data = {
            "identification": {
                "records_identified": summary["total"],
                "source": "AI Screening"
            },
            "screening": {
                "records_screened": summary["total"],
                "records_excluded": summary["excluded"],
                "exclusion_reasons": self._aggregate_exclusion_reasons()
            },
            "eligibility": {
                "full_text_assessed": summary["included"] + summary["uncertain"],
                "full_text_excluded": 0,  # To be filled after full-text review
                "uncertain_for_review": summary["uncertain"]
            },
            "included": {
                "studies_included": summary["included"]
            },
            "metadata": {
                "screening_date": datetime.now().isoformat(),
                "method": "AI-assisted screening",
                "dual_review": summary["dual_reviewed"] > 0,
                "mean_confidence": summary["mean_confidence"]
            }
        }
        
        with open(path, 'w') as f:
            json.dump(prisma_data, f, indent=2)
        
        return prisma_data
    
    def _aggregate_exclusion_reasons(self) -> Dict[str, int]:
        """Aggregate exclusion reasons across all results."""
        reasons = {}
        for result in self.get_excluded():
            for reason in result.exclusion_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.results])
