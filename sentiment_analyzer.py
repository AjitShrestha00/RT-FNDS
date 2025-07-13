import torch
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple dimensions and context awareness"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced sentiment dictionaries
        self.emotional_lexicon = {
            'positive_emotions': {
                'joy': ['happy', 'excited', 'thrilled', 'delighted', 'ecstatic', 'elated'],
                'hope': ['hopeful', 'optimistic', 'promising', 'encouraging', 'bright'],
                'trust': ['reliable', 'trustworthy', 'credible', 'authentic', 'genuine'],
                'surprise': ['amazing', 'incredible', 'astonishing', 'remarkable', 'extraordinary']
            },
            'negative_emotions': {
                'anger': ['angry', 'furious', 'outraged', 'enraged', 'livid', 'irate'],
                'fear': ['scared', 'terrified', 'panicked', 'anxious', 'worried', 'concerned'],
                'disgust': ['disgusting', 'revolting', 'appalling', 'horrible', 'awful'],
                'sadness': ['sad', 'depressed', 'melancholy', 'grief', 'sorrow', 'despair'],
                'contempt': ['contemptuous', 'disdainful', 'scornful', 'derisive']
            },
            'neutral_emotions': {
                'curiosity': ['curious', 'interested', 'intrigued', 'fascinated'],
                'confusion': ['confused', 'puzzled', 'perplexed', 'bewildered'],
                'indifference': ['neutral', 'indifferent', 'unconcerned', 'apathetic']
            }
        }
        
        # Fake news sentiment patterns
        self.fake_news_sentiment_patterns = {
            'sensationalism': ['shocking', 'incredible', 'unbelievable', 'mind-blowing', 'earth-shattering'],
            'urgency': ['urgent', 'breaking', 'exclusive', 'just in', 'developing', 'live'],
            'authority_claims': ['official', 'confirmed', 'verified', 'authentic', 'legitimate'],
            'emotional_manipulation': ['outrageous', 'scandalous', 'controversial', 'explosive', 'bombshell'],
            'conspiracy_language': ['secret', 'hidden', 'covered up', 'suppressed', 'censored'],
            'clickbait': ['you won\'t believe', 'what happens next', 'the truth about', 'they don\'t want you to know']
        }
        
        # Real news sentiment patterns
        self.real_news_sentiment_patterns = {
            'balanced_reporting': ['according to', 'reported by', 'confirmed by', 'verified sources'],
            'measured_language': ['study shows', 'research indicates', 'evidence suggests', 'analysis reveals'],
            'attribution': ['said', 'stated', 'announced', 'reported', 'confirmed'],
            'factual_indicators': ['data shows', 'statistics indicate', 'survey reveals', 'official records']
        }
    
    def analyze_sentiment_multi_dimensional(self, text: str) -> Dict[str, any]:
        """Advanced multi-dimensional sentiment analysis"""
        
        try:
            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Emotional intensity analysis
            emotional_scores = self._analyze_emotional_intensity(text)
            
            # Context-aware sentiment
            context_sentiment = self._analyze_context_sentiment(text)
            
            # Fake news sentiment indicators
            fake_indicators = self._analyze_fake_news_sentiment(text)
            
            # Real news sentiment indicators
            real_indicators = self._analyze_real_news_sentiment(text)
            
            # Sentiment consistency analysis
            sentiment_consistency = self._analyze_sentiment_consistency(text)
            
            # Overall sentiment classification
            overall_sentiment = self._classify_overall_sentiment(
                vader_scores, textblob_polarity, emotional_scores, 
                fake_indicators, real_indicators
            )
            
            return {
                'overall_sentiment': overall_sentiment,
                'vader_scores': vader_scores,
                'textblob_scores': {
                    'polarity': textblob_polarity,
                    'subjectivity': textblob_subjectivity
                },
                'emotional_scores': emotional_scores,
                'context_sentiment': context_sentiment,
                'fake_news_indicators': fake_indicators,
                'real_news_indicators': real_indicators,
                'sentiment_consistency': sentiment_consistency,
                'confidence_score': self._calculate_sentiment_confidence(
                    vader_scores, textblob_polarity, emotional_scores,
                    fake_indicators, real_indicators
                )
            }
            
        except Exception as e:
            logger.warning(f"Sentiment analysis error: {e}")
            return {
                'overall_sentiment': 'NEUTRAL',
                'vader_scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0},
                'textblob_scores': {'polarity': 0.0, 'subjectivity': 0.5},
                'emotional_scores': {'intensity': 0.0, 'dominant_emotion': 'neutral'},
                'context_sentiment': 'neutral',
                'fake_news_indicators': {'score': 0.0, 'patterns': []},
                'real_news_indicators': {'score': 0.0, 'patterns': []},
                'sentiment_consistency': 0.5,
                'confidence_score': 0.5
            }
    
    def _analyze_emotional_intensity(self, text: str) -> Dict[str, any]:
        """Analyze emotional intensity and dominant emotions"""
        text_lower = text.lower()
        emotion_counts = {}
        
        # Count emotions from lexicon
        for category, emotions in self.emotional_lexicon.items():
            for emotion, words in emotions.items():
                count = sum(1 for word in words if word in text_lower)
                if count > 0:
                    emotion_counts[emotion] = count
        
        # Calculate intensity
        total_emotions = sum(emotion_counts.values())
        intensity = min(total_emotions / 10.0, 1.0)  # Normalize to 0-1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        return {
            'intensity': intensity,
            'dominant_emotion': dominant_emotion,
            'emotion_counts': emotion_counts,
            'total_emotions': total_emotions
        }
    
    def _analyze_context_sentiment(self, text: str) -> str:
        """Analyze context-aware sentiment"""
        text_lower = text.lower()
        
        # Check for reporting language (neutral/objective)
        reporting_indicators = ['reported', 'said', 'announced', 'confirmed', 'stated']
        if any(indicator in text_lower for indicator in reporting_indicators):
            return 'objective'
        
        # Check for opinion language (subjective)
        opinion_indicators = ['believe', 'think', 'feel', 'opinion', 'view']
        if any(indicator in text_lower for indicator in opinion_indicators):
            return 'subjective'
        
        # Check for factual language
        factual_indicators = ['study', 'research', 'data', 'evidence', 'statistics']
        if any(indicator in text_lower for indicator in factual_indicators):
            return 'factual'
        
        return 'neutral'
    
    def _analyze_fake_news_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment patterns typical of fake news"""
        text_lower = text.lower()
        patterns_found = []
        total_score = 0
        
        for pattern_type, words in self.fake_news_sentiment_patterns.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                patterns_found.append(f"{pattern_type}: {count} instances")
                total_score += count
        
        return {
            'score': min(total_score / 5.0, 1.0),  # Normalize to 0-1
            'patterns': patterns_found,
            'total_indicators': total_score
        }
    
    def _analyze_real_news_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment patterns typical of real news"""
        text_lower = text.lower()
        patterns_found = []
        total_score = 0
        
        for pattern_type, words in self.real_news_sentiment_patterns.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                patterns_found.append(f"{pattern_type}: {count} instances")
                total_score += count
        
        return {
            'score': min(total_score / 5.0, 1.0),  # Normalize to 0-1
            'patterns': patterns_found,
            'total_indicators': total_score
        }
    
    def _analyze_sentiment_consistency(self, text: str) -> float:
        """Analyze consistency of sentiment throughout the text"""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 1.0
        
        sentence_sentiments = []
        for sentence in sentences:
            if sentence.strip():
                vader_score = self.vader_analyzer.polarity_scores(sentence)
                sentence_sentiments.append(vader_score['compound'])
        
        if len(sentence_sentiments) < 2:
            return 1.0
        
        # Calculate variance in sentiment
        mean_sentiment = np.mean(sentence_sentiments)
        variance = np.var(sentence_sentiments)
        
        # Higher consistency = lower variance
        consistency = max(0, 1 - variance)
        return consistency
    
    def _classify_overall_sentiment(self, vader_scores: Dict, textblob_polarity: float, 
                                  emotional_scores: Dict, fake_indicators: Dict, 
                                  real_indicators: Dict) -> str:
        """Classify overall sentiment based on multiple factors"""
        
        # Weighted decision
        vader_weight = 0.3
        textblob_weight = 0.2
        emotional_weight = 0.2
        fake_weight = 0.15
        real_weight = 0.15
        
        # VADER sentiment
        vader_sentiment = 'POSITIVE' if vader_scores['compound'] > 0.1 else \
                         'NEGATIVE' if vader_scores['compound'] < -0.1 else 'NEUTRAL'
        
        # TextBlob sentiment
        textblob_sentiment = 'POSITIVE' if textblob_polarity > 0.1 else \
                            'NEGATIVE' if textblob_polarity < -0.1 else 'NEUTRAL'
        
        # Emotional sentiment
        emotional_sentiment = 'POSITIVE' if emotional_scores['dominant_emotion'] in ['joy', 'hope', 'trust', 'surprise'] else \
                             'NEGATIVE' if emotional_scores['dominant_emotion'] in ['anger', 'fear', 'disgust', 'sadness', 'contempt'] else 'NEUTRAL'
        
        # Fake news sentiment (negative weight)
        fake_sentiment = 'NEGATIVE' if fake_indicators['score'] > 0.5 else 'NEUTRAL'
        
        # Real news sentiment (positive weight)
        real_sentiment = 'POSITIVE' if real_indicators['score'] > 0.5 else 'NEUTRAL'
        
        # Calculate weighted score
        sentiment_scores = {
            'POSITIVE': 0,
            'NEGATIVE': 0,
            'NEUTRAL': 0
        }
        
        # Add weighted scores
        sentiment_scores[vader_sentiment] += vader_weight
        sentiment_scores[textblob_sentiment] += textblob_weight
        sentiment_scores[emotional_sentiment] += emotional_weight
        sentiment_scores[fake_sentiment] += fake_weight
        sentiment_scores[real_sentiment] += real_weight
        
        # Return dominant sentiment
        return max(sentiment_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_sentiment_confidence(self, vader_scores: Dict, textblob_polarity: float,
                                      emotional_scores: Dict, fake_indicators: Dict,
                                      real_indicators: Dict) -> float:
        """Calculate confidence in sentiment analysis"""
        
        # Factors that increase confidence
        confidence_factors = []
        
        # High emotional intensity
        if emotional_scores['intensity'] > 0.5:
            confidence_factors.append(0.8)
        
        # Strong VADER compound score
        if abs(vader_scores['compound']) > 0.5:
            confidence_factors.append(0.9)
        
        # Clear fake news indicators
        if fake_indicators['score'] > 0.7:
            confidence_factors.append(0.8)
        
        # Clear real news indicators
        if real_indicators['score'] > 0.7:
            confidence_factors.append(0.8)
        
        # Low subjectivity (more objective)
        if abs(textblob_polarity) < 0.3:
            confidence_factors.append(0.7)
        
        # Calculate average confidence
        if confidence_factors:
            return min(sum(confidence_factors) / len(confidence_factors), 1.0)
        else:
            return 0.5  # Default confidence

def main():
    """Test the sentiment analyzer with various examples"""
    
    analyzer = AdvancedSentimentAnalyzer()
    
    test_texts = [
        "I'm so happy and excited about this amazing news!",
        "This is absolutely terrible and disgusting news.",
        "The study shows that 75% of participants reported positive outcomes.",
        "BREAKING: You won't believe what happens next! SHOCKING revelations!",
        "According to official sources, the event was confirmed yesterday.",
        "I think this might be interesting but I'm not sure.",
        "The data indicates a significant improvement in performance metrics."
    ]
    
    print("="*80)
    print("SENTIMENT ANALYSIS TESTING")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        result = analyzer.analyze_sentiment_multi_dimensional(text)
        
        print(f"   Overall Sentiment: {result['overall_sentiment']}")
        print(f"   VADER Compound: {result['vader_scores']['compound']:.3f}")
        print(f"   TextBlob Polarity: {result['textblob_scores']['polarity']:.3f}")
        print(f"   Emotional Intensity: {result['emotional_scores']['intensity']:.3f}")
        print(f"   Context: {result['context_sentiment']}")
        print(f"   Confidence: {result['confidence_score']:.3f}")
        
        if result['fake_news_indicators']['patterns']:
            print(f"   Fake News Indicators: {', '.join(result['fake_news_indicators']['patterns'])}")
        
        if result['real_news_indicators']['patterns']:
            print(f"   Real News Indicators: {', '.join(result['real_news_indicators']['patterns'])}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 