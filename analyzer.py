"""
Enhanced Earnings Call Transcript Analyzer
Analyzes financial transcripts to extract insights, sentiment, and investment signals.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

import fitz  # PyMuPDF
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Add these imports at the top of analyzer.py
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InvestmentView(Enum):
    """Investment outlook categories."""
    BULLISH = "Bullish 🟢"
    NEUTRAL = "Neutral 🟡"
    BEARISH = "Bearish 🔴"


@dataclass
class FinancialMetrics:
    """Container for financial metrics."""
    revenue: str
    profit: str
    ebitda: str
    margins: str
    trends: List[str]


@dataclass
class SentimentScores:
    """Container for sentiment analysis results."""
    management_score: float
    management_label: str
    qa_score: float
    qa_label: str


@dataclass
class AnalysisResult:
    """Complete analysis result container."""
    financials: Dict[str, Any]
    commentary: Dict[str, Any]
    sentiment: Dict[str, Any]
    risks: List[str]
    investment_view: Dict[str, Any]


class ModelLoader:
    """Lazy loading singleton for ML models."""
    _instance = None
    _nlp = None
    _sentiment_analyzer = None
    _summarizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model successfully")
            except OSError:
                logger.warning("Downloading spaCy model...")
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    @property
    def sentiment_analyzer(self):
        """Lazy load VADER sentiment analyzer."""
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("Loaded VADER sentiment analyzer")
        return self._sentiment_analyzer
    
    @property
    def summarizer(self):
        """Lazy load T5 summarization model."""
        if self._summarizer is None:
            self._summarizer = pipeline(
                "summarization",
                model="t5-small",
                tokenizer="t5-small",
                framework="pt"
            )
            logger.info("Loaded T5 summarization model")
        return self._summarizer


class PDFExtractor:
    """Handles PDF text extraction."""
    
    @staticmethod
    def extract_text(pdf_file) -> Optional[str]:
        """
        Extracts text from a PDF file.
        
        Args:
            pdf_file: File-like object or path to PDF
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            
            if not text.strip():
                logger.warning("PDF appears to be empty or contains no extractable text")
                return None
                
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None


class TranscriptParser:
    """Parses and splits transcript sections."""
    
    # Common Q&A section headers
    QA_PATTERNS = [
        r"(?i)\n\s*Q\s*&\s*A\s*Session\s*\n",
        r"(?i)\n\s*Question\s*(?:and|&)\s*Answer\s*Session\s*\n",
        r"(?i)\n\s*Q\s*&\s*A\s*\n",
        r"(?i)\n\s*Questions?\s*\n",
    ]
    
    @classmethod
    def split_transcript(cls, text: str) -> Tuple[str, str]:
        """
        Splits transcript into management commentary and Q&A sections.
        
        Args:
            text: Full transcript text
            
        Returns:
            Tuple of (management_commentary, qa_section)
        """
        for pattern in cls.QA_PATTERNS:
            parts = re.split(pattern, text, maxsplit=1)
            if len(parts) > 1:
                logger.info("Successfully split transcript into commentary and Q&A")
                return parts[0], parts[1]
        
        logger.warning("No Q&A section detected, treating entire text as commentary")
        return text, ""


class FinancialAnalyzer:
    """Extracts financial metrics and trends."""
    
    # Improved regex patterns with more flexibility
    PATTERNS = {
        'revenue': r"(?:Revenue|Sales|Turnover)[\s\w]*?(?:of|was|at|stood at)?[\s]*?(₹|Rs\.?|USD|\$|INR)\s*([\d,]+\.?\d*)\s*(Crores?|Cr\.?|Million|Mn|Billion|Bn)",
        'profit': r"(?:(?:Net\s*)?Profit(?:\s*After\s*Tax)?|PAT|Net\s*Income)[\s\w]*?(?:of|was|at|stood at)?[\s]*?(₹|Rs\.?|USD|\$|INR)\s*([\d,]+\.?\d*)\s*(Crores?|Cr\.?|Million|Mn|Billion|Bn)",
        'ebitda': r"(?:EBITDA|Operating\s*Profit)[\s\w]*?(?:of|was|at|stood at)?[\s]*?(₹|Rs\.?|USD|\$|INR)\s*([\d,]+\.?\d*)\s*(Crores?|Cr\.?|Million|Mn|Billion|Bn)",
        'margins': r"(?:margin|EBITDA\s*margin|Operating\s*margin)[\s\w]*?(?:of|was|at)?[\s]*([\d\.]+)\s*%",
    }
    
    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """
        Extracts financial metrics from text.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of financial metrics
        """
        financials = {}
        
        for key, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key == 'margins':
                    financials[key] = f"{matches[0]}%"
                else:
                    financials[key] = f"{matches[0][0]} {matches[0][1]} {matches[0][2]}"
            else:
                financials[key] = "Not found"
        
        # Extract growth trends
        trend_pattern = r"([^.!?]*?(?:grew|declined|increased|decreased|up|down)\s+by\s+[\d\.]+%[^.!?]*?(?:YoY|QoQ|year-on-year|quarter-on-quarter)[^.!?]*?[.!?])"
        trends = re.findall(trend_pattern, text, re.IGNORECASE)
        financials['trends'] = list(set(t.strip() for t in trends))
        
        logger.info(f"Extracted {len([v for v in financials.values() if v != 'Not found'])} financial metrics")
        return financials


class CommentaryAnalyzer:
    """Analyzes management commentary for insights."""
    
    POSITIVE_KEYWORDS = [
        'strong', 'growth', 'achieved', 'pleased', 'robust', 'grew',
        'optimistic', 'confident', 'improved', 'expansion', 'record',
        'exceeded', 'momentum', 'opportunity', 'innovative'
    ]
    
    NEGATIVE_KEYWORDS = [
        'headwinds', 'challenging', 'concerns', 'slowdown', 'pressure',
        'decline', 'cautious', 'uncertainty', 'weak', 'disappointing',
        'difficult', 'volatile', 'adversely', 'loss'
    ]
    
    def __init__(self, models: ModelLoader):
        self.models = models
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes commentary for positive/negative signals and strategy.
        
        Args:
            text: Management commentary text
            
        Returns:
            Dictionary with positives, concerns, and summary
        """
        doc = self.models.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        
        positives = []
        concerns = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in sent_lower)
            neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in sent_lower)
            
            if pos_count > neg_count and pos_count > 0:
                positives.append(sent)
            elif neg_count > pos_count and neg_count > 0:
                concerns.append(sent)
        
        # Generate summary with truncation for model limits
        max_length = min(1024, len(text))
        summary_text = text[:max_length]
        
        try:
            summary = self.models.summarizer(
                summary_text,
                max_length=150,
                min_length=40,
                do_sample=False
            )[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            summary = "Summary unavailable due to processing error."
        
        logger.info(f"Found {len(positives)} positive and {len(concerns)} concerning statements")
        
        return {
            'positives': positives[:10],  # Limit to top 10
            'concerns': concerns[:10],
            'summary': summary
        }


class SentimentAnalyzer:
    """Analyzes sentiment of different transcript sections."""
    
    def __init__(self, models: ModelLoader):
        self.models = models
    
    def get_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Calculates sentiment score and label.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (score, label)
        """
        if not text or not text.strip():
            return 0.0, "N/A"
        
        score = self.models.sentiment_analyzer.polarity_scores(text)['compound']
        
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return score, label
    
    def analyze(self, mgmt_text: str, qa_text: str) -> Dict[str, Any]:
        """
        Analyzes sentiment of management vs Q&A sections.
        
        Args:
            mgmt_text: Management commentary
            qa_text: Q&A section text
            
        Returns:
            Dictionary with sentiment scores and labels
        """
        mgmt_score, mgmt_label = self.get_sentiment(mgmt_text)
        qa_score, qa_label = self.get_sentiment(qa_text)
        
        logger.info(f"Management sentiment: {mgmt_label} ({mgmt_score:.2f})")
        logger.info(f"Q&A sentiment: {qa_label} ({qa_score:.2f})")
        
        return {
            'management_score': round(mgmt_score, 3),
            'management_label': mgmt_label,
            'qa_score': round(qa_score, 3),
            'qa_label': qa_label
        }


class RiskDetector:
    """Detects risk-related content."""
    
    RISK_KEYWORDS = [
        'risk', 'debt', 'regulatory', 'investigation', 'litigation',
        'supply chain', 'volatility', 'uncertain', 'competition',
        'slowdown', 'compliance', 'lawsuit', 'default', 'covenant',
        'impairment', 'restructuring', 'disruption'
    ]
    
    def __init__(self, models: ModelLoader):
        self.models = models
    
    def find_risks(self, text: str) -> List[str]:
        """
        Finds sentences containing risk-related keywords.
        
        Args:
            text: Full transcript text
            
        Returns:
            List of risk-related sentences
        """
        doc = self.models.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        risk_sentences = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in self.RISK_KEYWORDS):
                risk_sentences.append(sent)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_risks = []
        for risk in risk_sentences:
            if risk not in seen:
                seen.add(risk)
                unique_risks.append(risk)
        
        logger.info(f"Identified {len(unique_risks)} risk-related statements")
        return unique_risks[:15]  # Limit to top 15


class InvestmentViewGenerator:
    """Generates overall investment recommendation."""
    
    @staticmethod
    def generate(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates investment view based on analysis results.
        
        Args:
            analysis_data: Complete analysis results
            
        Returns:
            Dictionary with verdict, score, and rationale
        """
        score = 0
        max_score = 9
        rationale_parts = []
        
        # Financial metrics found (2 points)
        fin = analysis_data['financials']
        metrics_found = sum(1 for k in ['revenue', 'profit', 'ebitda'] if fin.get(k) != "Not found")
        if metrics_found >= 2:
            score += 2
            rationale_parts.append(f"Key financials disclosed ({metrics_found}/3)")
        elif metrics_found == 1:
            score += 1
            rationale_parts.append("Limited financial disclosure")
        
        # Commentary balance (2 points)
        commentary = analysis_data['commentary']
        pos_count = len(commentary.get('positives', []))
        con_count = len(commentary.get('concerns', []))
        
        if pos_count > con_count * 1.5:
            score += 2
            rationale_parts.append("Predominantly positive commentary")
        elif pos_count > con_count:
            score += 1
            rationale_parts.append("Balanced commentary with positive bias")
        elif con_count > pos_count:
            rationale_parts.append("Concerns highlighted in commentary")
        
        # Management sentiment (2 points)
        sentiment = analysis_data['sentiment']
        mgmt_score = sentiment.get('management_score', 0)
        
        if mgmt_score > 0.2:
            score += 2
            rationale_parts.append("Very positive management tone")
        elif mgmt_score > 0.05:
            score += 1
            rationale_parts.append("Positive management tone")
        elif mgmt_score < -0.05:
            rationale_parts.append("Cautious management tone")
        
        # Q&A sentiment (2 points)
        qa_score = sentiment.get('qa_score', 0)
        
        if qa_score > 0.1:
            score += 2
            rationale_parts.append("Constructive Q&A interactions")
        elif qa_score > 0:
            score += 1
            rationale_parts.append("Neutral Q&A interactions")
        
        # Risk flags (1 point)
        risk_count = len(analysis_data.get('risks', []))
        if risk_count < 5:
            score += 1
            rationale_parts.append("Limited risk disclosures")
        elif risk_count > 10:
            rationale_parts.append("Significant risk factors mentioned")
        
        # Determine verdict
        score_pct = (score / max_score) * 100
        
        if score_pct >= 65:
            verdict = InvestmentView.BULLISH.value
        elif score_pct >= 40:
            verdict = InvestmentView.NEUTRAL.value
        else:
            verdict = InvestmentView.BEARISH.value
        
        logger.info(f"Investment view: {verdict} (Score: {score}/{max_score})")
        
        return {
            'verdict': verdict,
            'score': score,
            'max_score': max_score,
            'score_percentage': round(score_pct, 1),
            'rationale': " | ".join(rationale_parts) if rationale_parts else "Insufficient data for strong signal"
        }


class TranscriptAnalyzer:
    """Main analyzer orchestrating all analysis components."""
    
    def __init__(self):
        self.models = ModelLoader()
        self.commentary_analyzer = CommentaryAnalyzer(self.models)
        self.sentiment_analyzer = SentimentAnalyzer(self.models)
        self.risk_detector = RiskDetector(self.models)
    
    def analyze(self, pdf_file) -> Dict[str, Any]:
        """
        Runs complete analysis pipeline on a PDF file.
        
        Args:
            pdf_file: File-like object containing PDF
            
        Returns:
            Dictionary with complete analysis results or error
        """
        try:
            # Extract text
            full_text = PDFExtractor.extract_text(pdf_file)
            if not full_text:
                return {"error": "Could not extract text from PDF"}
            
            # Split sections
            mgmt_commentary, qa_section = TranscriptParser.split_transcript(full_text)
            
            # Run all analyses
            financials = FinancialAnalyzer.analyze(full_text)
            commentary = self.commentary_analyzer.analyze(mgmt_commentary)
            sentiment = self.sentiment_analyzer.analyze(mgmt_commentary, qa_section)
            risks = self.risk_detector.find_risks(full_text)
            
            # Combine results
            all_analysis = {
                "financials": financials,
                "commentary": commentary,
                "sentiment": sentiment,
                "risks": risks
            }
            
            # Generate investment view
            investment_view = InvestmentViewGenerator.generate(all_analysis)
            all_analysis['investment_view'] = investment_view
            
            logger.info("Analysis completed successfully")
            return all_analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}


# Main entry point
def analyze_transcript(pdf_file) -> Dict[str, Any]:
    """
    Convenience function to analyze an earnings call transcript.
    
    Args:
        pdf_file: File-like object containing the PDF transcript
        
    Returns:
        Dictionary containing complete analysis results
    """
    analyzer = TranscriptAnalyzer()
    return analyzer.analyze(pdf_file)


if __name__ == "__main__":
    # Example usage
    print("Earnings Call Transcript Analyzer")
    print("=" * 50)
    print("Use: result = analyze_transcript(pdf_file_object)")

class Forecaster:
    """
    Handles ML-based time-series forecasting using Prophet.
    """
    
    @staticmethod
    def create_forecast(csv_file) -> go.Figure:
        """
        Uses Prophet to forecast financial data from a CSV.
        
        Args:
            csv_file: A file-like object (from st.file_uploader)
            
        Returns:
            A Plotly figure with the forecast.
        """
        try:
            data_df = pd.read_csv(csv_file)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise ValueError(f"Could not read CSV file: {e}")

        # Validate columns
        if 'ds' not in data_df.columns or 'y' not in data_df.columns:
            logger.error("CSV missing required 'ds' or 'y' columns.")
            raise ValueError("CSV must have 'ds' (date) and 'y' (value) columns.")
            
        try:
            # Convert date column to datetime
            data_df['ds'] = pd.to_datetime(data_df['ds'])
        except Exception as e:
            logger.error(f"Failed to parse dates in 'ds' column: {e}")
            raise ValueError(f"Could not parse dates in 'ds' column. Ensure format is YYYY-MM-DD.")
        
        logger.info("Training Prophet model...")
        model = Prophet()
        model.fit(data_df)
        
        # Forecast 4 quarters into the future
        future = model.make_future_dataframe(periods=4, freq='Q')
        forecast = model.predict(future)
        
        logger.info("Generating forecast plot.")
        fig = plot_plotly(model, forecast)
        
        fig.update_layout(
            title="ML-Powered Financial Forecast (Prophet)",
            xaxis_title="Date",
            yaxis_title="Value",
            showlegend=True
        )
        
        return fig