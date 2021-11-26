from .base import FieldType, indexPredicates
from dedupe import predicates

from affinegap import normalizedAffineGapDistance as affineGap
from highered import CRFEditDistance
from simplecosine.cosine import CosineTextSimilarity

from typing import Optional

crfEd = CRFEditDistance()

# cristianc: the predicates not yet supported are marked with TODO in the experiment
base_predicates = (predicates.wholeFieldPredicate,
                   predicates.firstTokenPredicate,  # TODO
                   predicates.commonIntegerPredicate,
                   predicates.nearIntegersPredicate,  # TODO
                   predicates.firstIntegerPredicate,  # TODO
                   predicates.hundredIntegerPredicate,
                   predicates.hundredIntegersOddPredicate,  # TODO
                   predicates.alphaNumericPredicate,  # TODO
                   predicates.sameThreeCharStartPredicate,
                   predicates.sameFiveCharStartPredicate,
                   predicates.sameSevenCharStartPredicate,
                   predicates.commonTwoTokens,
                   predicates.commonThreeTokens,
                   predicates.fingerprint,  # TODO
                   predicates.oneGramFingerprint,
                   predicates.twoGramFingerprint,  # TODO
                   predicates.sortedAcronym
                   )


class BaseStringType(FieldType):
    type: Optional[str] = None
    _Predicate = predicates.StringPredicate

    def __init__(self, definition):
        super(BaseStringType, self).__init__(definition)

        # cristianc: we are intentionally not considering these in the experiment
        self.predicates += []


class ShortStringType(BaseStringType):
    type = "ShortString"

    _predicate_functions = (base_predicates +
                            (predicates.commonFourGram,
                             predicates.commonSixGram,
                             predicates.tokenFieldPredicate,  # TODO
                             predicates.suffixArray,
                             predicates.doubleMetaphone,
                             predicates.metaphoneToken  # TODO
                             ))

    # cristianc: we are intentionally not considering these in the experiment
    _index_predicates = [
        # predicates.TfidfNGramCanopyPredicate,
        # predicates.TfidfNGramSearchPredicate
    ]
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(ShortStringType, self).__init__(definition)

        if definition.get('crf', False) is True:
            self.comparator = crfEd
        else:
            self.comparator = affineGap


class StringType(ShortStringType):
    type = "String"

    # cristianc: we are intentionally not considering these in the experiment
    _index_predicates = [
        # predicates.TfidfNGramCanopyPredicate,
        # predicates.TfidfNGramSearchPredicate,
        # predicates.TfidfTextCanopyPredicate,
        # predicates.TfidfTextSearchPredicate
    ]


class TextType(BaseStringType):
    type = "Text"

    _predicate_functions = base_predicates

    # cristianc: we are intentionally not considering these in the experiment
    _index_predicates = [
        # predicates.TfidfTextCanopyPredicate,
        # predicates.TfidfTextSearchPredicate
    ]
    _index_thresholds = (0.2, 0.4, 0.6, 0.8)

    def __init__(self, definition):
        super(TextType, self).__init__(definition)

        if 'corpus' not in definition:
            definition['corpus'] = []

        self.comparator = CosineTextSimilarity(definition['corpus'])
