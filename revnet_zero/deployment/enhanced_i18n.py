"""
RevNet-Zero Enhanced Internationalization - Advanced multi-language support.
Provides comprehensive internationalization for global RevNet-Zero deployments.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"

class MessageCategory(Enum):
    """Categories of translatable messages"""
    ERRORS = "errors"
    WARNINGS = "warnings"  
    INFO = "info"
    API_RESPONSES = "api_responses"
    UI_LABELS = "ui_labels"
    HELP_TEXT = "help_text"
    VALIDATION_MESSAGES = "validation_messages"
    PERFORMANCE_METRICS = "performance_metrics"

@dataclass
class LocalizationContext:
    """Context for localization formatting"""
    language: Language
    region: str
    timezone: str
    currency: str = "USD"
    number_format: str = "en_US"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"

class EnhancedI18nManager:
    """
    Enhanced internationalization manager for RevNet-Zero.
    
    Features:
    - Multi-language message translation
    - Context-aware formatting
    - Regional customization
    - Dynamic language switching
    - Pluralization support
    - RTL language support
    """
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        
        # Translation dictionaries
        self.translations: Dict[Language, Dict[str, Dict[str, str]]] = {}
        
        # Context settings
        self.localization_contexts: Dict[Language, LocalizationContext] = {}
        
        # RTL language support
        self.rtl_languages = {Language.ARABIC}
        
        # Initialize with built-in translations
        self._initialize_translations()
        self._initialize_contexts()
    
    def _initialize_translations(self):
        """Initialize built-in translations"""
        
        # English (base language)
        self.translations[Language.ENGLISH] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "Required dependency '{dependency}' not found",
                "validation_failed": "Input validation failed: {reason}",
                "memory_limit_exceeded": "Memory limit exceeded: {current}MB > {limit}MB",
                "deployment_failed": "Deployment failed in region {region}: {error}",
                "compliance_violation": "Compliance violation detected: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "Successfully deployed to {region}",
                "cache_hit": "Cache hit for key: {key}",
                "performance_optimized": "Performance optimized for {component}",
                "compliance_validated": "Compliance validated for {standard}",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "Memory Usage",
                "cache_hit_rate": "Cache Hit Rate",
                "performance_score": "Performance Score",
                "compliance_status": "Compliance Status",
                "regional_health": "Regional Health",
                "global_status": "Global Status",
            }
        }
        
        # Spanish translations
        self.translations[Language.SPANISH] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "Dependencia requerida '{dependency}' no encontrada",
                "validation_failed": "Validación de entrada falló: {reason}",
                "memory_limit_exceeded": "Límite de memoria excedido: {current}MB > {limit}MB",
                "deployment_failed": "Despliegue falló en región {region}: {error}",
                "compliance_violation": "Violación de cumplimiento detectada: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "Desplegado exitosamente en {region}",
                "cache_hit": "Acierto de caché para clave: {key}",
                "performance_optimized": "Rendimiento optimizado para {component}",
                "compliance_validated": "Cumplimiento validado para {standard}",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "Uso de Memoria",
                "cache_hit_rate": "Tasa de Aciertos de Caché",
                "performance_score": "Puntuación de Rendimiento",
                "compliance_status": "Estado de Cumplimiento",
                "regional_health": "Salud Regional",
                "global_status": "Estado Global",
            }
        }
        
        # French translations
        self.translations[Language.FRENCH] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "Dépendance requise '{dependency}' introuvable",
                "validation_failed": "Échec de la validation d'entrée: {reason}",
                "memory_limit_exceeded": "Limite mémoire dépassée: {current}MB > {limit}MB",
                "deployment_failed": "Échec du déploiement dans la région {region}: {error}",
                "compliance_violation": "Violation de conformité détectée: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "Déployé avec succès dans {region}",
                "cache_hit": "Succès de cache pour clé: {key}",
                "performance_optimized": "Performance optimisée pour {component}",
                "compliance_validated": "Conformité validée pour {standard}",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "Utilisation Mémoire",
                "cache_hit_rate": "Taux de Réussite Cache",
                "performance_score": "Score de Performance",
                "compliance_status": "Statut de Conformité",
                "regional_health": "Santé Régionale",
                "global_status": "Statut Global",
            }
        }
        
        # German translations
        self.translations[Language.GERMAN] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "Erforderliche Abhängigkeit '{dependency}' nicht gefunden",
                "validation_failed": "Eingabevalidierung fehlgeschlagen: {reason}",
                "memory_limit_exceeded": "Speicherlimit überschritten: {current}MB > {limit}MB",
                "deployment_failed": "Bereitstellung in Region {region} fehlgeschlagen: {error}",
                "compliance_violation": "Compliance-Verletzung erkannt: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "Erfolgreich in {region} bereitgestellt",
                "cache_hit": "Cache-Treffer für Schlüssel: {key}",
                "performance_optimized": "Leistung optimiert für {component}",
                "compliance_validated": "Compliance validiert für {standard}",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "Speicherverbrauch",
                "cache_hit_rate": "Cache-Trefferrate",
                "performance_score": "Leistungsbewertung",
                "compliance_status": "Compliance-Status",
                "regional_health": "Regionale Gesundheit",
                "global_status": "Globaler Status",
            }
        }
        
        # Japanese translations
        self.translations[Language.JAPANESE] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "必要な依存関係 '{dependency}' が見つかりません",
                "validation_failed": "入力検証に失敗しました: {reason}",
                "memory_limit_exceeded": "メモリ制限を超過: {current}MB > {limit}MB",
                "deployment_failed": "リージョン {region} でのデプロイに失敗: {error}",
                "compliance_violation": "コンプライアンス違反が検出されました: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "{region} への正常なデプロイが完了",
                "cache_hit": "キー {key} のキャッシュヒット",
                "performance_optimized": "{component} のパフォーマンスが最適化されました",
                "compliance_validated": "{standard} のコンプライアンスが検証されました",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "メモリ使用量",
                "cache_hit_rate": "キャッシュヒット率",
                "performance_score": "パフォーマンススコア",
                "compliance_status": "コンプライアンス状態",
                "regional_health": "リージョナルヘルス",
                "global_status": "グローバル状態",
            }
        }
        
        # Chinese Simplified
        self.translations[Language.CHINESE_SIMPLIFIED] = {
            MessageCategory.ERRORS.value: {
                "dependency_not_found": "未找到所需依赖项 '{dependency}'",
                "validation_failed": "输入验证失败: {reason}",
                "memory_limit_exceeded": "内存限制超出: {current}MB > {limit}MB",
                "deployment_failed": "在区域 {region} 部署失败: {error}",
                "compliance_violation": "检测到合规性违规: {standard}",
            },
            MessageCategory.INFO.value: {
                "deployment_success": "成功部署到 {region}",
                "cache_hit": "缓存键 {key} 命中",
                "performance_optimized": "{component} 性能已优化",
                "compliance_validated": "{standard} 合规性已验证",
            },
            MessageCategory.UI_LABELS.value: {
                "memory_usage": "内存使用量",
                "cache_hit_rate": "缓存命中率",
                "performance_score": "性能评分",
                "compliance_status": "合规状态",
                "regional_health": "区域健康状况",
                "global_status": "全局状态",
            }
        }
    
    def _initialize_contexts(self):
        """Initialize localization contexts"""
        
        self.localization_contexts = {
            Language.ENGLISH: LocalizationContext(
                language=Language.ENGLISH,
                region="US",
                timezone="America/New_York",
                currency="USD",
                number_format="en_US",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p"
            ),
            Language.SPANISH: LocalizationContext(
                language=Language.SPANISH,
                region="ES",
                timezone="Europe/Madrid",
                currency="EUR",
                number_format="es_ES",
                date_format="%d/%m/%Y",
                time_format="%H:%M"
            ),
            Language.FRENCH: LocalizationContext(
                language=Language.FRENCH,
                region="FR",
                timezone="Europe/Paris",
                currency="EUR",
                number_format="fr_FR",
                date_format="%d/%m/%Y",
                time_format="%H:%M"
            ),
            Language.GERMAN: LocalizationContext(
                language=Language.GERMAN,
                region="DE",
                timezone="Europe/Berlin",
                currency="EUR",
                number_format="de_DE",
                date_format="%d.%m.%Y",
                time_format="%H:%M"
            ),
            Language.JAPANESE: LocalizationContext(
                language=Language.JAPANESE,
                region="JP",
                timezone="Asia/Tokyo",
                currency="JPY",
                number_format="ja_JP",
                date_format="%Y/%m/%d",
                time_format="%H:%M"
            ),
            Language.CHINESE_SIMPLIFIED: LocalizationContext(
                language=Language.CHINESE_SIMPLIFIED,
                region="CN",
                timezone="Asia/Shanghai",
                currency="CNY",
                number_format="zh_CN",
                date_format="%Y年%m月%d日",
                time_format="%H:%M"
            ),
        }
    
    def set_language(self, language: Language):
        """Set current language"""
        self.current_language = language
    
    def get_text(self, key: str, category: MessageCategory = MessageCategory.INFO,
                language: Optional[Language] = None, **kwargs) -> str:
        """Get translated text with parameter substitution"""
        
        target_language = language or self.current_language
        
        # Get translation
        if (target_language in self.translations and
            category.value in self.translations[target_language] and
            key in self.translations[target_language][category.value]):
            
            text = self.translations[target_language][category.value][key]
        else:
            # Fallback to English
            if (self.default_language in self.translations and
                category.value in self.translations[self.default_language] and
                key in self.translations[self.default_language][category.value]):
                
                text = self.translations[self.default_language][category.value][key]
            else:
                # Ultimate fallback
                text = f"[{key}]"
                warnings.warn(f"Translation not found: {key} in {category.value}", UserWarning)
        
        # Parameter substitution
        try:
            return text.format(**kwargs)
        except KeyError as e:
            warnings.warn(f"Missing parameter for translation {key}: {e}", UserWarning)
            return text
    
    def format_number(self, number: Union[int, float], 
                     language: Optional[Language] = None) -> str:
        """Format number according to locale"""
        
        target_language = language or self.current_language
        context = self.localization_contexts.get(target_language)
        
        if not context:
            return str(number)
        
        # Simple formatting based on locale
        if context.number_format.startswith("en"):
            return f"{number:,.2f}" if isinstance(number, float) else f"{number:,}"
        elif context.number_format.startswith("de"):
            # German uses . for thousands and , for decimals
            if isinstance(number, float):
                return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"{number:,}".replace(",", ".")
        elif context.number_format.startswith("fr"):
            # French uses space for thousands and , for decimals
            if isinstance(number, float):
                return f"{number:,.2f}".replace(",", " ").replace(".", ",")
            else:
                return f"{number:,}".replace(",", " ")
        else:
            return str(number)
    
    def format_percentage(self, value: float, 
                         language: Optional[Language] = None) -> str:
        """Format percentage according to locale"""
        
        formatted_number = self.format_number(value, language)
        return f"{formatted_number}%"
    
    def format_currency(self, amount: float, 
                       language: Optional[Language] = None) -> str:
        """Format currency according to locale"""
        
        target_language = language or self.current_language
        context = self.localization_contexts.get(target_language)
        
        if not context:
            return f"${amount:.2f}"
        
        formatted_amount = self.format_number(amount, language)
        
        # Currency symbol placement
        if context.currency == "USD":
            return f"${formatted_amount}"
        elif context.currency == "EUR":
            return f"{formatted_amount} €"
        elif context.currency == "JPY":
            return f"¥{formatted_amount}"
        elif context.currency == "CNY":
            return f"¥{formatted_amount}"
        else:
            return f"{formatted_amount} {context.currency}"
    
    def is_rtl_language(self, language: Optional[Language] = None) -> bool:
        """Check if language is right-to-left"""
        target_language = language or self.current_language
        return target_language in self.rtl_languages
    
    def get_available_languages(self) -> List[Language]:
        """Get list of available languages"""
        return list(self.translations.keys())
    
    def add_translation(self, language: Language, category: MessageCategory,
                       key: str, text: str):
        """Add or update translation"""
        
        if language not in self.translations:
            self.translations[language] = {}
        
        if category.value not in self.translations[language]:
            self.translations[language][category.value] = {}
        
        self.translations[language][category.value][key] = text

# Global i18n manager instance
_enhanced_i18n_manager = None

def get_enhanced_i18n_manager() -> EnhancedI18nManager:
    """Get global enhanced i18n manager instance"""
    global _enhanced_i18n_manager
    if _enhanced_i18n_manager is None:
        _enhanced_i18n_manager = EnhancedI18nManager()
    return _enhanced_i18n_manager

def translate(key: str, category: MessageCategory = MessageCategory.INFO, **kwargs) -> str:
    """Convenience function for getting translated text"""
    return get_enhanced_i18n_manager().get_text(key, category, **kwargs)

def set_global_language(language: Language):
    """Set global language"""
    get_enhanced_i18n_manager().set_language(language)

def get_current_language() -> Language:
    """Get current language"""
    return get_enhanced_i18n_manager().current_language

def localized_number(number: Union[int, float]) -> str:
    """Format number using current locale"""
    return get_enhanced_i18n_manager().format_number(number)

def localized_percentage(value: float) -> str:
    """Format percentage using current locale"""
    return get_enhanced_i18n_manager().format_percentage(value)

def localized_currency(amount: float) -> str:
    """Format currency using current locale"""
    return get_enhanced_i18n_manager().format_currency(amount)

__all__ = [
    'Language',
    'MessageCategory',
    'LocalizationContext',
    'EnhancedI18nManager',
    'get_enhanced_i18n_manager',
    'translate',
    'set_global_language',
    'get_current_language',
    'localized_number',
    'localized_percentage',
    'localized_currency'
]