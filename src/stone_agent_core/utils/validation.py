from typing import Dict, Any, Type, Optional, List, TypeVar
from dataclasses import is_dataclass, fields
from enum import Enum
import inspect

T = TypeVar('T')

class ValidationError(Exception):
    """Base class for validation errors."""
    pass

class InvalidStateError(ValidationError):
    """Raised when the agent state is invalid."""
    pass

class InvalidConfigError(ValidationError):
    """Raised when a configuration is invalid."""
    pass

def validate_state(state: Dict[str, Any], state_class: Type) -> None:
    """
    Validate that a state dictionary matches the expected state class structure.
    
    Args:
        state: The state dictionary to validate
        state_class: The expected state class (should be a TypedDict or dataclass)
        
    Raises:
        InvalidStateError: If the state is invalid
    """
    if not isinstance(state, dict):
        raise InvalidStateError(f"Expected state to be a dict, got {type(state).__name__}")
    
    # Handle TypedDict
    if hasattr(state_class, '__annotations__'):
        required_fields = state_class.__annotations__.keys()
        for field in required_fields:
            if field not in state:
                raise InvalidStateError(f"Missing required field in state: {field}")
    
    # Handle dataclasses
    elif is_dataclass(state_class):
        for field in fields(state_class):
            if field.name not in state and field.default is None and field.default_factory is None:
                raise InvalidStateError(f"Missing required field in state: {field.name}")

def validate_config(config: Dict[str, Any], config_class: Type[T]) -> T:
    """
    Validate and convert a configuration dictionary to a config object.
    
    Args:
        config: The configuration dictionary to validate
        config_class: The configuration class to validate against
        
    Returns:
        An instance of config_class with the validated configuration
        
    Raises:
        InvalidConfigError: If the configuration is invalid
    """
    if not isinstance(config, dict):
        raise InvalidConfigError(f"Expected config to be a dict, got {type(config).__name__}")
    
    try:
        # For dataclasses
        if is_dataclass(config_class):
            return config_class(**config)
        # For TypedDict
        elif hasattr(config_class, '__annotations__'):
            return config_class(**config)  # type: ignore
        else:
            raise InvalidConfigError(f"Unsupported config class type: {config_class.__name__}")
    except (TypeError, ValueError) as e:
        raise InvalidConfigError(f"Invalid configuration: {str(e)}")

def validate_enum(value: Any, enum_class: Type[Enum]) -> bool:
    """
    Validate that a value is a valid enum value.
    
    Args:
        value: The value to validate
        enum_class: The enum class to validate against
        
    Returns:
        bool: True if the value is valid, False otherwise
    """
    try:
        if value in enum_class.__members__.values():
            return True
        if str(value) in enum_class.__members__:
            return True
        return False
    except (TypeError, ValueError):
        return False

def validate_required_fields(obj: Any, required_fields: List[str]) -> None:
    """
    Validate that required fields are present in an object.
    
    Args:
        obj: The object to validate
        required_fields: List of required field names
        
    Raises:
        ValueError: If any required fields are missing
    """
    missing = [field for field in required_fields if not hasattr(obj, field) or getattr(obj, field) is None]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")