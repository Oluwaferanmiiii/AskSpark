import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import json
import time
from .ai_providers import UnifiedAIClient, ModelResponse
from ..config.settings import Config
from ..config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ComparisonMetrics:
    """Metrics for model comparison"""
    model: str
    provider: str
    response_time: float
    tokens_used: int
    cost: float
    quality_score: float
    relevance_score: float
    completeness_score: float

class ModelComparisonEngine:
    """Engine for comparing AI models across multiple metrics"""
    
    def __init__(self):
        self.client = UnifiedAIClient()
        self.comparison_history = []
        
    def _calculate_quality_scores(self, response: str, expected_keywords: List[str] = None) -> Dict[str, float]:
        """Calculate quality metrics for a response"""
        # Basic quality metrics
        word_count = len(response.split())
        sentence_count = response.count('.') + response.count('!') + response.count('?')
        
        # Relevance score (based on keyword presence)
        relevance_score = 0.5  # Base score
        if expected_keywords:
            found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
            relevance_score = min(1.0, 0.3 + (found_keywords / len(expected_keywords)) * 0.7)
        
        # Completeness score (based on length and structure)
        completeness_score = min(1.0, word_count / 100)  # Normalize to 100 words as full score
        
        # Quality score (combination of factors)
        quality_score = (relevance_score * 0.5 + completeness_score * 0.3 + min(1.0, sentence_count / 10) * 0.2)
        
        return {
            'quality_score': round(quality_score, 3),
            'relevance_score': round(relevance_score, 3),
            'completeness_score': round(completeness_score, 3)
        }
    
    def compare_models(self, prompt: str, providers_models: List[Tuple[str, str]], 
                      expected_keywords: List[str] = None) -> List[ComparisonMetrics]:
        """Compare multiple models on the same prompt"""
        messages = [{"role": "user", "content": prompt}]
        results = []
        
        for provider, model in providers_models:
            try:
                logger.info(f"Testing {provider}/{model}")
                response = self.client.call_model(provider, model, messages)
                
                # Calculate quality scores
                scores = self._calculate_quality_scores(response.content, expected_keywords)
                
                metrics = ComparisonMetrics(
                    model=model,
                    provider=provider,
                    response_time=response.response_time,
                    tokens_used=response.tokens_used or 0,
                    cost=response.cost or 0.0,
                    **scores
                )
                
                results.append(metrics)
                logger.info(f"Completed {provider}/{model}: Quality={scores['quality_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error testing {provider}/{model}: {str(e)}")
                continue
        
        self.comparison_history.extend(results)
        return results
    
    def benchmark_models(self, test_prompts: List[str], providers_models: List[Tuple[str, str]]) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple prompts"""
        all_results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Running benchmark prompt {i+1}/{len(test_prompts)}")
            results = self.compare_models(prompt, providers_models)
            all_results.extend(results)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(result) for result in all_results])
        
        # Calculate aggregate metrics
        summary = df.groupby(['provider', 'model']).agg({
            'response_time': ['mean', 'std'],
            'tokens_used': ['mean', 'std'],
            'cost': ['mean', 'std'],
            'quality_score': ['mean', 'std'],
            'relevance_score': ['mean', 'std'],
            'completeness_score': ['mean', 'std']
        }).round(3)
        
        return df, summary
    
    def get_recommendations(self, results: List[ComparisonMetrics], 
                          priority: str = 'balanced') -> List[Dict]:
        """Get model recommendations based on different priorities"""
        if not results:
            return []
        
        df = pd.DataFrame([asdict(result) for result in results])
        
        recommendations = []
        
        # Best overall (balanced approach)
        if priority == 'balanced':
            df['balanced_score'] = (
                df['quality_score'] * 0.4 +
                (1 - df['cost'] / df['cost'].max()) * 0.3 +
                (1 - df['response_time'] / df['response_time'].max()) * 0.3
            )
            best = df.loc[df['balanced_score'].idxmax()]
            recommendations.append({
                'category': 'Best Overall',
                'model': best['model'],
                'provider': best['provider'],
                'score': round(best['balanced_score'], 3),
                'reasoning': 'Balanced performance across quality, cost, and speed'
            })
        
        # Fastest response
        fastest = df.loc[df['response_time'].idxmin()]
        recommendations.append({
            'category': 'Fastest Response',
            'model': fastest['model'],
            'provider': fastest['provider'],
            'score': round(fastest['response_time'], 3),
            'reasoning': f'Lowest response time at {fastest["response_time"]:.2f}s'
        })
        
        # Highest quality
        highest_quality = df.loc[df['quality_score'].idxmax()]
        recommendations.append({
            'category': 'Highest Quality',
            'model': highest_quality['model'],
            'provider': highest_quality['provider'],
            'score': round(highest_quality['quality_score'], 3),
            'reasoning': 'Best response quality and relevance'
        })
        
        # Most cost-effective
        most_cost_effective = df.loc[df['cost'].idxmin()]
        recommendations.append({
            'category': 'Most Cost-Effective',
            'model': most_cost_effective['model'],
            'provider': most_cost_effective['provider'],
            'score': round(most_cost_effective['cost'], 6),
            'reasoning': f'Lowest cost at ${most_cost_effective["cost"]:.6f}'
        })
        
        return recommendations
    
    def create_comparison_visualization(self, results: List[ComparisonMetrics]) -> go.Figure:
        """Create interactive comparison dashboard"""
        df = pd.DataFrame([asdict(result) for result in results])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time Comparison', 'Cost Comparison', 
                          'Quality Scores', 'Performance Overview'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Response Time
        fig.add_trace(
            go.Bar(x=df['model'], y=df['response_time'], name='Response Time (s)',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Cost
        fig.add_trace(
            go.Bar(x=df['model'], y=df['cost'], name='Cost ($)',
                   marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Quality Scores
        fig.add_trace(
            go.Scatter(x=df['model'], y=df['quality_score'], mode='markers+lines',
                      name='Quality Score', marker_size=10, marker_color='red'),
            row=2, col=1
        )
        
        # Performance Overview (bubble chart)
        fig.add_trace(
            go.Scatter(
                x=df['response_time'],
                y=df['quality_score'],
                mode='markers',
                marker=dict(
                    size=df['cost'] * 10000,  # Scale cost for bubble size
                    color=df['provider'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df['model'],
                name='Performance (size=cost)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Model Performance Comparison Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def export_results(self, results: List[ComparisonMetrics], filename: str = 'comparison_results.json'):
        """Export comparison results to JSON"""
        data = {
            'timestamp': time.time(),
            'results': [asdict(result) for result in results],
            'recommendations': self.get_recommendations(results)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {filename}")
        return filename
