"""
Internationalization (i18n) support for RevNet-Zero global deployment.

Provides multi-language support, localization utilities, and 
compliance features for international markets.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum


logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class RegionCompliance(Enum):
    """Compliance frameworks for different regions."""
    GDPR = "gdpr"          # European Union
    CCPA = "ccpa"          # California, USA
    PDPA = "pdpa"          # Singapore
    PIPEDA = "pipeda"      # Canada
    LGPD = "lgpd"          # Brazil


class InternationalizationManager:
    """Manager for internationalization and localization."""
    
    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.compliance_settings = {}
        self._load_translations()
        self._setup_compliance()
    
    def _load_translations(self):
        """Load translation files for supported languages."""
        translations_dir = Path(__file__).parent / "translations"
        
        # Create default translations if directory doesn't exist
        if not translations_dir.exists():
            translations_dir.mkdir(exist_ok=True)
            self._create_default_translations(translations_dir)
        
        # Load existing translations
        for lang in SupportedLanguage:
            lang_file = translations_dir / f"{lang.value}.json"
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang.value] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load translations for {lang.value}: {e}")
                    self.translations[lang.value] = {}
            else:
                self.translations[lang.value] = {}
    
    def _create_default_translations(self, translations_dir: Path):
        """Create default translation files."""
        
        # English (base translations)
        en_translations = {
            "errors": {
                "invalid_input": "Invalid input provided",
                "processing_failed": "Processing failed",
                "insufficient_memory": "Insufficient memory available",
                "model_load_failed": "Failed to load model",
                "timeout_exceeded": "Operation timeout exceeded"
            },
            "messages": {
                "training_started": "Training started",
                "training_completed": "Training completed successfully",
                "model_saved": "Model saved successfully",
                "validation_passed": "Validation passed",
                "processing_batch": "Processing batch {batch_num} of {total_batches}"
            },
            "ui": {
                "welcome": "Welcome to RevNet-Zero",
                "loading": "Loading...",
                "progress": "Progress",
                "settings": "Settings",
                "help": "Help",
                "about": "About"
            },
            "compliance": {
                "data_processing_notice": "We process your data in accordance with applicable privacy laws",
                "consent_required": "Your consent is required for this operation",
                "data_retention_policy": "Data is retained according to our retention policy"
            }
        }
        
        # Spanish translations
        es_translations = {
            "errors": {
                "invalid_input": "Entrada inválida proporcionada",
                "processing_failed": "El procesamiento falló",
                "insufficient_memory": "Memoria insuficiente disponible",
                "model_load_failed": "Error al cargar el modelo",
                "timeout_exceeded": "Tiempo de operación excedido"
            },
            "messages": {
                "training_started": "Entrenamiento iniciado",
                "training_completed": "Entrenamiento completado exitosamente",
                "model_saved": "Modelo guardado exitosamente",
                "validation_passed": "Validación aprobada",
                "processing_batch": "Procesando lote {batch_num} de {total_batches}"
            },
            "ui": {
                "welcome": "Bienvenido a RevNet-Zero",
                "loading": "Cargando...",
                "progress": "Progreso",
                "settings": "Configuración",
                "help": "Ayuda",
                "about": "Acerca de"
            },
            "compliance": {
                "data_processing_notice": "Procesamos sus datos de acuerdo con las leyes de privacidad aplicables",
                "consent_required": "Se requiere su consentimiento para esta operación",
                "data_retention_policy": "Los datos se conservan según nuestra política de retención"
            }
        }
        
        # French translations
        fr_translations = {
            "errors": {
                "invalid_input": "Entrée invalide fournie",
                "processing_failed": "Le traitement a échoué",
                "insufficient_memory": "Mémoire insuffisante disponible",
                "model_load_failed": "Échec du chargement du modèle",
                "timeout_exceeded": "Délai d'opération dépassé"
            },
            "messages": {
                "training_started": "Entraînement commencé",
                "training_completed": "Entraînement terminé avec succès",
                "model_saved": "Modèle sauvegardé avec succès",
                "validation_passed": "Validation réussie",
                "processing_batch": "Traitement du lot {batch_num} sur {total_batches}"
            },
            "ui": {
                "welcome": "Bienvenue à RevNet-Zero",
                "loading": "Chargement...",
                "progress": "Progrès",
                "settings": "Paramètres",
                "help": "Aide",
                "about": "À propos"
            },
            "compliance": {
                "data_processing_notice": "Nous traitons vos données conformément aux lois de confidentialité applicables",
                "consent_required": "Votre consentement est requis pour cette opération",
                "data_retention_policy": "Les données sont conservées selon notre politique de rétention"
            }
        }
        
        # German translations
        de_translations = {
            "errors": {
                "invalid_input": "Ungültige Eingabe bereitgestellt",
                "processing_failed": "Verarbeitung fehlgeschlagen",
                "insufficient_memory": "Unzureichender Arbeitsspeicher verfügbar",
                "model_load_failed": "Laden des Modells fehlgeschlagen",
                "timeout_exceeded": "Operationszeitlimit überschritten"
            },
            "messages": {
                "training_started": "Training gestartet",
                "training_completed": "Training erfolgreich abgeschlossen",
                "model_saved": "Modell erfolgreich gespeichert",
                "validation_passed": "Validierung bestanden",
                "processing_batch": "Verarbeitung Batch {batch_num} von {total_batches}"
            },
            "ui": {
                "welcome": "Willkommen bei RevNet-Zero",
                "loading": "Lädt...",
                "progress": "Fortschritt",
                "settings": "Einstellungen",
                "help": "Hilfe",
                "about": "Über"
            },
            "compliance": {
                "data_processing_notice": "Wir verarbeiten Ihre Daten gemäß den geltenden Datenschutzgesetzen",
                "consent_required": "Ihre Zustimmung ist für diesen Vorgang erforderlich",
                "data_retention_policy": "Daten werden gemäß unserer Aufbewahrungsrichtlinie gespeichert"
            }
        }
        
        # Japanese translations
        ja_translations = {
            "errors": {
                "invalid_input": "無効な入力が提供されました",
                "processing_failed": "処理に失敗しました",
                "insufficient_memory": "利用可能なメモリが不足しています",
                "model_load_failed": "モデルの読み込みに失敗しました",
                "timeout_exceeded": "操作のタイムアウトを超過しました"
            },
            "messages": {
                "training_started": "トレーニングが開始されました",
                "training_completed": "トレーニングが正常に完了しました",
                "model_saved": "モデルが正常に保存されました",
                "validation_passed": "検証に合格しました",
                "processing_batch": "バッチ {batch_num} / {total_batches} を処理中"
            },
            "ui": {
                "welcome": "RevNet-Zeroへようこそ",
                "loading": "読み込み中...",
                "progress": "進行状況",
                "settings": "設定",
                "help": "ヘルプ",
                "about": "概要"
            },
            "compliance": {
                "data_processing_notice": "適用されるプライバシー法に従ってデータを処理します",
                "consent_required": "この操作にはあなたの同意が必要です",
                "data_retention_policy": "データは当社の保持ポリシーに従って保持されます"
            }
        }
        
        # Chinese translations
        zh_translations = {
            "errors": {
                "invalid_input": "提供的输入无效",
                "processing_failed": "处理失败",
                "insufficient_memory": "可用内存不足",
                "model_load_failed": "模型加载失败",
                "timeout_exceeded": "操作超时"
            },
            "messages": {
                "training_started": "训练已开始",
                "training_completed": "训练成功完成",
                "model_saved": "模型保存成功",
                "validation_passed": "验证通过",
                "processing_batch": "正在处理批次 {batch_num} / {total_batches}"
            },
            "ui": {
                "welcome": "欢迎使用 RevNet-Zero",
                "loading": "加载中...",
                "progress": "进度",
                "settings": "设置",
                "help": "帮助",
                "about": "关于"
            },
            "compliance": {
                "data_processing_notice": "我们根据适用的隐私法律处理您的数据",
                "consent_required": "此操作需要您的同意",
                "data_retention_policy": "数据根据我们的保留政策进行保留"
            }
        }
        
        # Save translation files
        translations_map = {
            "en": en_translations,
            "es": es_translations,
            "fr": fr_translations,
            "de": de_translations,
            "ja": ja_translations,
            "zh": zh_translations
        }
        
        for lang_code, translations in translations_map.items():
            lang_file = translations_dir / f"{lang_code}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created default translations for {len(translations_map)} languages")
    
    def _setup_compliance(self):
        """Setup compliance settings for different regions."""
        self.compliance_settings = {
            RegionCompliance.GDPR.value: {
                "requires_explicit_consent": True,
                "data_retention_days": 365,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required": True  # Data Protection Officer
            },
            RegionCompliance.CCPA.value: {
                "requires_explicit_consent": True,
                "data_retention_days": 365,
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "non_discrimination": True
            },
            RegionCompliance.PDPA.value: {
                "requires_explicit_consent": True,
                "data_retention_days": 365,
                "purpose_limitation": True,
                "data_minimization": True,
                "accuracy_requirement": True
            },
            RegionCompliance.PIPEDA.value: {
                "requires_explicit_consent": True,
                "data_retention_days": 365,
                "purpose_limitation": True,
                "openness_principle": True,
                "individual_access": True
            },
            RegionCompliance.LGPD.value: {
                "requires_explicit_consent": True,
                "data_retention_days": 365,
                "purpose_limitation": True,
                "data_minimization": True,
                "transparency": True
            }
        }
    
    def set_language(self, language_code: str):
        """Set the current language for the application."""
        if language_code in [lang.value for lang in SupportedLanguage]:
            self.current_language = language_code
            logger.info(f"Language set to: {language_code}")
        else:
            logger.warning(f"Unsupported language: {language_code}, using default: {self.default_language}")
            self.current_language = self.default_language
    
    def get_text(self, key_path: str, **kwargs) -> str:
        """Get localized text for the current language."""
        keys = key_path.split('.')
        
        # Try current language first
        translations = self.translations.get(self.current_language, {})
        text = self._get_nested_value(translations, keys)
        
        # Fallback to default language if not found
        if text is None and self.current_language != self.default_language:
            translations = self.translations.get(self.default_language, {})
            text = self._get_nested_value(translations, keys)
        
        # Final fallback to the key itself
        if text is None:
            text = key_path
            logger.warning(f"Translation not found for key: {key_path}")
        
        # Format with provided arguments
        try:
            if kwargs:
                text = text.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to format translation '{key_path}': {e}")
        
        return text
    
    def _get_nested_value(self, dictionary: Dict[str, Any], keys: List[str]) -> Optional[str]:
        """Get nested value from dictionary using dot notation keys."""
        current = dictionary
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current if isinstance(current, str) else None
    
    def get_compliance_requirements(self, region: str) -> Dict[str, Any]:
        """Get compliance requirements for a specific region."""
        return self.compliance_settings.get(region, {})
    
    def is_consent_required(self, region: str) -> bool:
        """Check if explicit consent is required for a region."""
        requirements = self.get_compliance_requirements(region)
        return requirements.get("requires_explicit_consent", False)
    
    def get_data_retention_days(self, region: str) -> int:
        """Get data retention period for a region."""
        requirements = self.get_compliance_requirements(region)
        return requirements.get("data_retention_days", 365)
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable name for a language code."""
        language_names = {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh": "中文"
        }
        return language_names.get(language_code, language_code)


class ComplianceValidator:
    """Validator for regional compliance requirements."""
    
    def __init__(self, i18n_manager: InternationalizationManager):
        self.i18n = i18n_manager
    
    def validate_data_processing(self, region: str, has_consent: bool, data_type: str) -> Dict[str, Any]:
        """Validate if data processing is compliant for a region."""
        requirements = self.i18n.get_compliance_requirements(region)
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "requirements": requirements
        }
        
        # Check consent requirement
        if requirements.get("requires_explicit_consent", False) and not has_consent:
            validation_result["compliant"] = False
            validation_result["violations"].append({
                "type": "missing_consent",
                "message": self.i18n.get_text("compliance.consent_required")
            })
        
        # Additional checks based on data type
        if data_type == "personal_data":
            if region == RegionCompliance.GDPR.value and not requirements.get("privacy_by_design", False):
                validation_result["violations"].append({
                    "type": "privacy_by_design",
                    "message": "GDPR requires privacy by design implementation"
                })
        
        return validation_result
    
    def get_retention_policy(self, region: str) -> Dict[str, Any]:
        """Get data retention policy for a region."""
        retention_days = self.i18n.get_data_retention_days(region)
        
        return {
            "retention_days": retention_days,
            "policy_text": self.i18n.get_text("compliance.data_retention_policy"),
            "automatic_deletion": True,
            "user_requested_deletion": True
        }


# Global instance
_i18n_manager = None


def get_i18n_manager() -> InternationalizationManager:
    """Get global internationalization manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = InternationalizationManager()
    return _i18n_manager


def set_language(language_code: str):
    """Set global language."""
    get_i18n_manager().set_language(language_code)


def get_text(key_path: str, **kwargs) -> str:
    """Get localized text (convenience function)."""
    return get_i18n_manager().get_text(key_path, **kwargs)


def is_region_compliant(region: str, has_consent: bool = True, data_type: str = "general") -> bool:
    """Check if current setup is compliant with region requirements."""
    validator = ComplianceValidator(get_i18n_manager())
    result = validator.validate_data_processing(region, has_consent, data_type)
    return result["compliant"]