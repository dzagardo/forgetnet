# forgetnet/core.py

from abc import ABC, abstractmethod

class PrivacyMechanism(ABC):
    @abstractmethod
    def apply(self, weights):
        """Apply the privacy mechanism to the given weights."""
        pass

    @abstractmethod
    def get_privacy_spent(self):
        """Return the privacy budget spent."""
        pass