from abc import ABC, abstractmethod

class CVXLayer(ABC):

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass