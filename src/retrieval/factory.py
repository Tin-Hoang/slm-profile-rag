"""Factory for creating retrieval strategies."""

import logging
from typing import Any

from .base import BaseRetrieverStrategy

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating and managing retrieval strategies.

    This factory uses a registry pattern to allow dynamic registration
    of new retrieval strategies. Strategies are registered using the
    @RetrieverFactory.register decorator.

    Example:
        @RetrieverFactory.register("my_strategy")
        class MyStrategy(BaseRetrieverStrategy):
            ...

        # Later:
        strategy = RetrieverFactory.create("my_strategy", config)
    """

    _strategies: dict[str, type[BaseRetrieverStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a retrieval strategy.

        Args:
            name: Unique identifier for the strategy

        Returns:
            Decorator function that registers the strategy class

        Example:
            @RetrieverFactory.register("vector")
            class VectorStrategy(BaseRetrieverStrategy):
                ...
        """

        def decorator(strategy_class: type[BaseRetrieverStrategy]):
            if name in cls._strategies:
                logger.warning(f"Overwriting existing strategy: {name}")
            cls._strategies[name] = strategy_class
            logger.debug(f"Registered retrieval strategy: {name}")
            return strategy_class

        return decorator

    @classmethod
    def create(cls, strategy_name: str, config: dict[str, Any]) -> BaseRetrieverStrategy:
        """Create a retrieval strategy instance by name.

        Args:
            strategy_name: Name of the strategy to create
            config: Configuration dictionary for the strategy

        Returns:
            An instance of the requested strategy

        Raises:
            ValueError: If strategy_name is not registered
        """
        if strategy_name not in cls._strategies:
            available = ", ".join(cls._strategies.keys()) or "none"
            msg = f"Unknown retrieval strategy: '{strategy_name}'. Available: {available}"
            raise ValueError(msg)

        strategy_class = cls._strategies[strategy_name]
        logger.info(f"Creating retrieval strategy: {strategy_name}")

        return strategy_class(config)

    @classmethod
    def available_strategies(cls) -> list[str]:
        """List all registered strategy names.

        Returns:
            List of registered strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy name to check

        Returns:
            True if strategy is registered, False otherwise
        """
        return name in cls._strategies

    @classmethod
    def get_strategy_class(cls, name: str) -> type[BaseRetrieverStrategy] | None:
        """Get the strategy class by name without instantiating.

        Args:
            name: Strategy name

        Returns:
            Strategy class or None if not found
        """
        return cls._strategies.get(name)


def get_retriever_factory() -> type[RetrieverFactory]:
    """Get the RetrieverFactory class.

    Returns:
        The RetrieverFactory class
    """
    return RetrieverFactory
