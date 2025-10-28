from mikecore.DfsuFile import DfsuFile, DfsuFileType

from .dfsu3dsigma import Dfsu3DSigma
from .dfsuverticalprofilesigma import DfsuVerticalProfileSigma

def read(filename, unit_conversion=1):
    ds = DfsuFile.Open(filename)
    if ds.DfsuFileType == DfsuFileType.Dfsu3DSigma:
        dfsu = Dfsu3DSigma(ds, unit_conversion=unit_conversion)
    elif ds.DfsuFileType == DfsuFileType.DfsuVerticalProfileSigma:
        dfsu = DfsuVerticalProfileSigma(ds, unit_conversion=unit_conversion)
    else:
        raise ValueError("Unsupported dfsu file type. Only Dfsu3DSigma and DfsuVerticalProfileSigma are supported.")
    return dfsu



if __name__ == "__main__":
    pass