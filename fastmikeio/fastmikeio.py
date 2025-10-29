from mikecore.DfsuFile import DfsuFile, DfsuFileType

from .dfsu import Dfsu3DSigma, DfsuVerticalProfileSigma

def read(filename, unit_conversion=1):
    ds = DfsuFile.Open(filename)
    if ds.DfsuFileType == DfsuFileType.Dfsu3DSigma:
        dfsu = Dfsu3DSigma(filename, unit_conversion=unit_conversion)
    elif ds.DfsuFileType == DfsuFileType.DfsuVerticalProfileSigma:
        dfsu = DfsuVerticalProfileSigma(filename, unit_conversion=unit_conversion)
    else:
        raise ValueError("Unsupported dfsu file type. Only Dfsu3DSigma and DfsuVerticalProfileSigma are supported.")
    return dfsu



if __name__ == "__main__":
    pass