import abc

from typing import TypeVar, Generic

VisitingT = TypeVar("VisitingT")
VisitorT = TypeVar("VisitorT")
VisitingArgsT = TypeVar("VisitingArgsT")
MatcherT = TypeVar("MatcherT")


class MatchingVisitor(Generic[VisitingT, VisitingArgsT], abc.ABC):

    @abc.abstractmethod
    async def visit(self, other: VisitingT, args: VisitingArgsT):
        pass


