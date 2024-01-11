import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

t_data = pd.read_csv("./Train_datasets/Train-1542865627584.csv")
train_bendata = pd.read_csv("./Train_datasets/Train_Beneficiarydata-1542865627584.csv")
train_inpdata = pd.read_csv("./Train_datasets/Train_Inpatientdata-1542865627584.csv")
train_outpdata = pd.read_csv("./Train_datasets/Train_Outpatientdata-1542865627584.csv")

te_data = pd.read_csv("./Test_datasets/Test-1542969243754.csv")
test_bendata = pd.read_csv("./Test_datasets/Test_Beneficiarydata-1542969243754.csv")
test_inpdata = pd.read_csv("./Test_datasets/Test_Inpatientdata-1542969243754.csv")
test_outpdata = pd.read_csv("./Test_datasets/Test_Outpatientdata-1542969243754.csv")

st.title("Fraudulent Heathcare Insurance Providers")

st.sidebar.header(" Upload Here")
st.sidebar.markdown("---")
df1, df2, df3, df4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
file = st.sidebar.file_uploader(
    "Upload Test-1542969243754 dataset here", type="csv", key="1"
)
if file is not None:
    df1 = pd.read_csv(file)
    if list(df1.columns) != list(te_data.columns):
        st.write("Please upload the required dataset")
if len(df1) > 0:
    file1 = st.sidebar.file_uploader(
        "Upload Test_Beneficiarydata-1542969243754 dataset here", type="csv", key="2"
    )
    if file1 is not None:
        df2 = pd.read_csv(file1)
        if list(df2.columns) != list(test_bendata.columns):
            st.write("Please upload the required dataset")
    if len(df2) > 0:
        file2 = st.sidebar.file_uploader(
            "Upload Test_Inpatientdata-1542969243754 dataset here", type="csv", key="3"
        )
        if file2 is not None:
            df3 = pd.read_csv(file2)
            if list(df3.columns) != list(test_inpdata.columns):
                st.write("Please upload the required dataset")
        if len(df3) > 0:
            file3 = st.sidebar.file_uploader(
                "Upload Test_Outpatientdata-1542969243754 dataset here",
                type="csv",
                key="4",
            )
            if file3 is not None:
                df4 = pd.read_csv(file3)
                if list(df4.columns) != list(test_outpdata.columns):
                    st.write("Please upload the required dataset")
            if len(df4) > 0:

                test_data = df1
                test_bendata = df2
                test_inpdata = df3
                test_outpdata = df4

                df = t_data.merge(train_outpdata, how="left", on="Provider")
                df = df.merge(
                    train_inpdata, how="left", on=["Provider", "BeneID", "ClaimID"]
                )
                df = df.merge(train_bendata, how="left", on="BeneID")

                df = df[~df["BeneID"].isna()]
                df = df.drop(
                    [
                        "ClaimStartDt_y",
                        "ClaimEndDt_y",
                        "InscClaimAmtReimbursed_y",
                        "AttendingPhysician_y",
                        "OperatingPhysician_y",
                        "OtherPhysician_y",
                        "ClmAdmitDiagnosisCode_y",
                        "DeductibleAmtPaid_y",
                        "ClmDiagnosisCode_1_y",
                        "ClmDiagnosisCode_2_y",
                        "ClmDiagnosisCode_3_y",
                        "ClmDiagnosisCode_4_y",
                        "ClmDiagnosisCode_5_y",
                        "ClmDiagnosisCode_6_y",
                        "ClmDiagnosisCode_7_y",
                        "ClmDiagnosisCode_8_y",
                        "ClmDiagnosisCode_9_y",
                        "ClmDiagnosisCode_10_y",
                        "ClmProcedureCode_1_y",
                        "ClmProcedureCode_2_y",
                        "ClmProcedureCode_3_y",
                        "ClmProcedureCode_4_y",
                        "ClmProcedureCode_5_y",
                        "ClmProcedureCode_6_y",
                        "AdmissionDt",
                        "DischargeDt",
                        "DeductibleAmtPaid_y",
                        "DiagnosisGroupCode",
                    ],
                    axis=1,
                )
                df[
                    [
                        "ClmDiagnosisCode_1_x",
                        "ClmDiagnosisCode_2_x",
                        "ClmDiagnosisCode_3_x",
                        "ClmDiagnosisCode_4_x",
                        "ClmDiagnosisCode_5_x",
                        "ClmDiagnosisCode_6_x",
                        "ClmDiagnosisCode_7_x",
                        "ClmDiagnosisCode_8_x",
                        "ClmDiagnosisCode_9_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                ] = df[
                    [
                        "ClmDiagnosisCode_1_x",
                        "ClmDiagnosisCode_2_x",
                        "ClmDiagnosisCode_3_x",
                        "ClmDiagnosisCode_4_x",
                        "ClmDiagnosisCode_5_x",
                        "ClmDiagnosisCode_6_x",
                        "ClmDiagnosisCode_7_x",
                        "ClmDiagnosisCode_8_x",
                        "ClmDiagnosisCode_9_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                ].replace(
                    np.nan, 0
                )

                df[
                    [
                        "ClmProcedureCode_1_x",
                        "ClmProcedureCode_2_x",
                        "ClmProcedureCode_3_x",
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ]
                ] = df[
                    [
                        "ClmProcedureCode_1_x",
                        "ClmProcedureCode_2_x",
                        "ClmProcedureCode_3_x",
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ]
                ].replace(
                    np.nan, 0
                )
                df["ClmAdmitDiagnosisCode_x"] = df["ClmAdmitDiagnosisCode_x"].replace(
                    np.nan, 0
                )

                df[
                    ["AttendingPhysician_x", "OperatingPhysician_x", "OtherPhysician_x"]
                ] = df[
                    ["AttendingPhysician_x", "OperatingPhysician_x", "OtherPhysician_x"]
                ].replace(
                    np.nan, "NA"
                )

                lst = [
                    "ChronicCond_Alzheimer",
                    "ChronicCond_Heartfailure",
                    "ChronicCond_KidneyDisease",
                    "ChronicCond_Cancer",
                    "ChronicCond_ObstrPulmonary",
                    "ChronicCond_Depression",
                    "ChronicCond_Diabetes",
                    "ChronicCond_IschemicHeart",
                    "ChronicCond_Osteoporasis",
                    "ChronicCond_rheumatoidarthritis",
                    "ChronicCond_stroke",
                ]

                df["DOB"] = pd.to_datetime(df["DOB"], format="%Y-%m-%d")
                df["DOD"] = pd.to_datetime(df["DOD"], format="%Y-%m-%d")
                df["DOD"].fillna(df["DOD"].max(), inplace=True)

                df["ClaimStartDt_x"] = pd.to_datetime(
                    df["ClaimStartDt_x"], format="%Y-%m-%d"
                )
                df["ClaimEndDt_x"] = pd.to_datetime(
                    df["ClaimEndDt_x"], format="%Y-%m-%d"
                )
                df["Claim_Duration"] = (
                    df["ClaimEndDt_x"] - df["ClaimStartDt_x"]
                ).dt.days

                df["B_Age"] = round(((df["DOD"] - df["DOB"]).dt.days) / 365, 1)

                # df is the dataframe, a='ClaimID',x= Attending Physician, y=Operating Physician, z=Other Physician

                def AllPhy_totClaims(df, a, x, y, z):
                    df["AttPhy_TC"] = df.groupby(x)[a].transform("count")
                    df["OptPhy_TC"] = df.groupby(y)[a].transform("count")
                    df["OthPhy_TC"] = df.groupby(z)[a].transform("count")
                    df["AttPhy_TC"].fillna(0, inplace=True)
                    df["OptPhy_TC"].fillna(0, inplace=True)
                    df["OthPhy_TC"].fillna(0, inplace=True)
                    df["AllPhy_totClaims"] = (
                        df["AttPhy_TC"] + df["OptPhy_TC"] + df["OthPhy_TC"]
                    )
                    df.drop(
                        ["AttPhy_TC", "OptPhy_TC", "OthPhy_TC"], axis=1, inplace=True
                    )
                    return df

                AllPhy_totClaims(
                    df,
                    "ClaimID",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                )

                # Provider and all physicians interaction

                # df is dataframe, a=Provider, x=Attending Physician, y=Operating Physician, z= Other Physician

                def Prvd_AllPhy(df, a, x, y, z):
                    df["Prvd_AttPhy"] = df.groupby(a)[x].transform("count")
                    df["Prvd_OptPhy"] = df.groupby(a)[y].transform("count")
                    df["Prvd_OthPhy"] = df.groupby(a)[z].transform("count")
                    df["Prvd_AllPhy"] = (
                        df["Prvd_AttPhy"] + df["Prvd_OptPhy"] + df["Prvd_OthPhy"]
                    )
                    df.drop(
                        ["Prvd_AttPhy", "Prvd_OptPhy", "Prvd_OthPhy"],
                        axis=1,
                        inplace=True,
                    )
                    return df

                Prvd_AllPhy(
                    df,
                    "Provider",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                )

                # Unique claim diagnosis used by providers

                # df is dataframe, a=provider, x= claimadmitdiagnosiscode,

                def Pvrd_CADC(df, a, x):
                    df["Pvrd_CADC"] = df.groupby(a)[x].transform("nunique")
                    return df

                Pvrd_CADC(df, "Provider", "ClmAdmitDiagnosisCode_x")

                # Sum of Insc Claim Re-Imb Amount for a Provider

                # df is dataframe, a= provider,x=Insurance claim amt reinmbursed

                def Pvrd_InsClnReimAmt(df, a, x):
                    df["Pvrd_InsClnReimAmt"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_InsClnReimAmt(df, "Provider", "InscClaimAmtReimbursed_x")

                df["RenalDiseaseIndicator"].replace({0: 0, "Y": 1}, inplace=True)
                df["RenalDiseaseIndicator"] = df["RenalDiseaseIndicator"].astype("int")

                # renal disease indicator seen by provider
                # df is dataframe, a= provider,x=renal disease indicator
                def Pvrd_RDI(df, a, x):
                    df["Pvrd_RDI"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_RDI(df, "Provider", "RenalDiseaseIndicator")

                # df is dataframe, a= provider,x=deductible amount paid
                def Pvrd_DedAmtPaid(df, a, x):
                    df["Pvrd_DedAmtPaid"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_DedAmtPaid(df, "Provider", "DeductibleAmtPaid_x")

                # df is dataframe, a= provider,x= IP annual amt reimbursement
                def Pvrd_IPAnAmtReim(df, a, x):
                    df["Pvrd_IPAnAmtReim"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_IPAnAmtReim(df, "Provider", "IPAnnualReimbursementAmt")

                # df is dataframe, a= provider,x= OP annual amt reimbursement
                def Pvrd_OPAnAmtReim(df, a, x):
                    df["Pvrd_OPAnAmtReim"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_OPAnAmtReim(df, "Provider", "OPAnnualReimbursementAmt")

                # df is dataframe, a= provider,x= OP annual deductible amt
                def Pvrd_OPAnDedAmt(df, a, x):
                    df["Pvrd_OPAnDedAmt"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_OPAnDedAmt(df, "Provider", "OPAnnualDeductibleAmt")

                # df is dataframe, a= provider,x= IP annual deductible amt
                def Pvrd_IPAnDedAmt(df, a, x):
                    df["Pvrd_IPAnDedAmt"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_IPAnDedAmt(df, "Provider", "IPAnnualDeductibleAmt")

                # df is dataframe, a= provider,x=Claim duration
                def Pvrd_ClmDur(df, a, x):
                    df["Pvrd_ClmDur"] = df.groupby(a)[x].transform("sum")
                    return df

                Pvrd_ClmDur(df, "Provider", "Claim_Duration")

                def create_agg_feats(df, grp_col, feat_name, operation="sum"):
                    feat_1 = feat_name + "_Insc_ReImb_Amt"
                    df[feat_1] = df.groupby(grp_col)[
                        "InscClaimAmtReimbursed_x"
                    ].transform(operation)

                    feat_2 = feat_name + "_DedAmtPaid"
                    df[feat_2] = df.groupby(grp_col)["DeductibleAmtPaid_x"].transform(
                        operation
                    )

                    feat_3 = feat_name + "_IP_Annual_ReImb_Amt"
                    df[feat_3] = df.groupby(grp_col)[
                        "IPAnnualReimbursementAmt"
                    ].transform(operation)

                    feat_4 = feat_name + "_IP_Annual_Ded_Amt"
                    df[feat_4] = df.groupby(grp_col)["IPAnnualDeductibleAmt"].transform(
                        operation
                    )

                    feat_5 = feat_name + "_OP_Annual_ReImb_Amt"
                    df[feat_5] = df.groupby(grp_col)[
                        "OPAnnualReimbursementAmt"
                    ].transform(operation)

                    feat_6 = feat_name + "_OP_Annual_Ded_Amt"
                    df[feat_6] = df.groupby(grp_col)["OPAnnualDeductibleAmt"].transform(
                        operation
                    )

                    feat_7 = feat_name + "_Claim_Duration"
                    df[feat_7] = df.groupby(grp_col)["Claim_Duration"].transform(
                        operation
                    )

                create_agg_feats(df, grp_col="BeneID", feat_name="BENE")
                create_agg_feats(
                    df, grp_col="AttendingPhysician_x", feat_name="ATT_PHY"
                )
                create_agg_feats(
                    df, grp_col="OperatingPhysician_x", feat_name="OPT_PHY"
                )
                create_agg_feats(df, grp_col="OtherPhysician_x", feat_name="OTH_PHY")
                create_agg_feats(
                    df,
                    grp_col="ClmAdmitDiagnosisCode_x",
                    feat_name="Claim_Admit_Diag_Code",
                )

                df.drop(
                    [
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ],
                    axis=1,
                    inplace=True,
                )

                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_1_x", feat_name="Claim_DiagCode1"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_2_x", feat_name="Claim_DiagCode2"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_3_x", feat_name="Claim_DiagCode3"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_4_x", feat_name="Claim_DiagCode4"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_5_x", feat_name="Claim_DiagCode5"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_6_x", feat_name="Claim_DiagCode6"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_7_x", feat_name="Claim_DiagCode7"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_8_x", feat_name="Claim_DiagCode8"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_9_x", feat_name="Claim_DiagCode9"
                )
                create_agg_feats(
                    df, grp_col="ClmDiagnosisCode_10_x", feat_name="Claim_DiagCode10"
                )

                create_agg_feats(
                    df, grp_col="ClmProcedureCode_1_x", feat_name="Claim_ProcCode1"
                )
                create_agg_feats(
                    df, grp_col="ClmProcedureCode_2_x", feat_name="Claim_ProcCode2"
                )
                create_agg_feats(
                    df, grp_col="ClmProcedureCode_3_x", feat_name="Claim_ProcCode3"
                )

                # PROVIDER <--> other features :: To get claim counts

                df["ClmCount_Provider"] = df.groupby(["Provider"])["ClaimID"].transform(
                    "count"
                )
                df["ClmCount_Provider_BeneID"] = df.groupby(["Provider", "BeneID"])[
                    "ClaimID"
                ].transform("count")
                df["ClmCount_Provider_AttendingPhysician"] = df.groupby(
                    ["Provider", "AttendingPhysician_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_OtherPhysician"] = df.groupby(
                    ["Provider", "OtherPhysician_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_OperatingPhysician"] = df.groupby(
                    ["Provider", "OperatingPhysician_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmAdmitDiagnosisCode"] = df.groupby(
                    ["Provider", "ClmAdmitDiagnosisCode_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmProcedureCode_1"] = df.groupby(
                    ["Provider", "ClmProcedureCode_1_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmProcedureCode_2"] = df.groupby(
                    ["Provider", "ClmProcedureCode_2_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmProcedureCode_3"] = df.groupby(
                    ["Provider", "ClmProcedureCode_3_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_1"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_1_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_2"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_2_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_3"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_3_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_4"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_4_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_5"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_5_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_6"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_6_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_7"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_7_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_8"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_8_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_9"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_9_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_ClmDiagnosisCode_10"] = df.groupby(
                    ["Provider", "ClmDiagnosisCode_10_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> ATTENDING PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OPERATING PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OTHER PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_1"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_1_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_2"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_2_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_3"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_3_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> ATTENDING PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_4"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_4_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_5"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_5_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_6"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_6_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_7"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_7_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_8"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_8_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_9"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_9_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_10"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OPERATING PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_4"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_4_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_5"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_5_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_6"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_6_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_7"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_7_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_8"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_8_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_9"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_9_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_10"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OTHER PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_1"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_1_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_2"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_2_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_3"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_3_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_4"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_4_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_5"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_5_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_6"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_6_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_7"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_7_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_8"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_8_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_9"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_9_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_10"
                ] = df.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_10_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> PROCEDURE CODES :: To get claim counts
                df["ClmCount_Provider_BeneID_ClmProcedureCode_1"] = df.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_1_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmProcedureCode_2"] = df.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_2_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmProcedureCode_3"] = df.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_3_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> DIAGNOSIS CODES :: To get claim counts
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_1_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_2"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_2_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_3"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_3_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_4"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_4_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_5"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_5_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_6"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_6_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_7"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_7_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_8"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_8_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_9"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_9_x"]
                )["ClaimID"].transform("count")
                df["ClmCount_Provider_BeneID_ClmDiagnosisCode_10"] = df.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_10_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> DIAGNOSIS CODES <--> PROCEDURE CODES :: To get claim counts
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_1"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_2"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_3"
                ] = df.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                columns_to_remove = [
                    "BeneID",
                    "ClaimID",
                    "ClaimStartDt_x",
                    "ClaimEndDt_x",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                    "ClmAdmitDiagnosisCode_x",
                    "ClmDiagnosisCode_1_x",
                    "ClmDiagnosisCode_2_x",
                    "ClmDiagnosisCode_3_x",
                    "ClmDiagnosisCode_4_x",
                    "ClmDiagnosisCode_5_x",
                    "ClmDiagnosisCode_6_x",
                    "ClmDiagnosisCode_7_x",
                    "ClmDiagnosisCode_8_x",
                    "ClmDiagnosisCode_9_x",
                    "ClmDiagnosisCode_10_x",
                    "ClmProcedureCode_1_x",
                    "ClmProcedureCode_2_x",
                    "ClmProcedureCode_3_x",
                    "DOB",
                    "DOD",
                    "State",
                    "County",
                ]
                df.drop(columns_to_remove, axis=1, inplace=True)

                df["Gender"].replace({2.0: 0, 1.0: 1}, inplace=True)
                df["PotentialFraud"].replace({"No": 0, "Yes": 1}, inplace=True)
                df["PotentialFraud"] = df["PotentialFraud"].astype("int")
                df["ChronicCond_Alzheimer"].replace(2.0, 0, inplace=True)
                df["ChronicCond_Heartfailure"].replace(2.0, 0, inplace=True)
                df["ChronicCond_KidneyDisease"].replace(2.0, 0, inplace=True)
                df["ChronicCond_Cancer"].replace(2.0, 0, inplace=True)
                df["ChronicCond_ObstrPulmonary"].replace(2.0, 0, inplace=True)
                df["ChronicCond_Depression"].replace(2.0, 0, inplace=True)
                df["ChronicCond_Diabetes"].replace(2.0, 0, inplace=True)
                df["ChronicCond_IschemicHeart"].replace(2.0, 0, inplace=True)
                df["ChronicCond_Osteoporasis"].replace(2.0, 0, inplace=True)
                df["ChronicCond_rheumatoidarthritis"].replace(2.0, 0, inplace=True)
                df["ChronicCond_stroke"].replace(2.0, 0, inplace=True)

                df = pd.get_dummies(df, columns=["Race"])

                df = df.groupby(["Provider", "PotentialFraud"], as_index=False).agg(
                    "sum"
                )

                from sklearn.model_selection import train_test_split

                x = df.iloc[:, 2:]
                y = df["PotentialFraud"]

                stdscl = StandardScaler()
                x = stdscl.fit_transform(x)

                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, stratify=y, test_size=0.2, random_state=0
                )

                # # XGBoost

                import xgboost as xgb

                xclf = xgb.XGBClassifier(
                    learning_rate=0.15, max_depth=5, n_estimators=300
                )
                xclf.fit(x_train, y_train)
                bn = xclf.predict(x_test)

                imp_columns = pd.DataFrame(
                    {
                        "Features": df.columns[2:],
                        "Importance_Model_1": xclf.feature_importances_,
                    }
                )

                top_15 = imp_columns.sort_values(
                    by="Importance_Model_1", ascending=False
                ).head(15)

                p = sns.barplot(
                    data=top_15, y=top_15["Features"], x=top_15["Importance_Model_1"]
                )
                plt.bar_label(p.containers[0])
                plt.show()

                df_t = test_data.merge(test_outpdata, how="left", on="Provider")
                df_t = df_t.merge(
                    test_inpdata, how="left", on=["Provider", "BeneID", "ClaimID"]
                )
                df_t = df_t.merge(test_bendata, how="left", on="BeneID")

                df_t = df_t[~df_t["BeneID"].isna()]

                df_t = df_t.drop(
                    [
                        "ClaimStartDt_y",
                        "ClaimEndDt_y",
                        "InscClaimAmtReimbursed_y",
                        "AttendingPhysician_y",
                        "OperatingPhysician_y",
                        "OtherPhysician_y",
                        "ClmAdmitDiagnosisCode_y",
                        "DeductibleAmtPaid_y",
                        "ClmDiagnosisCode_1_y",
                        "ClmDiagnosisCode_2_y",
                        "ClmDiagnosisCode_3_y",
                        "ClmDiagnosisCode_4_y",
                        "ClmDiagnosisCode_5_y",
                        "ClmDiagnosisCode_6_y",
                        "ClmDiagnosisCode_7_y",
                        "ClmDiagnosisCode_8_y",
                        "ClmDiagnosisCode_9_y",
                        "ClmDiagnosisCode_10_y",
                        "ClmProcedureCode_1_y",
                        "ClmProcedureCode_2_y",
                        "ClmProcedureCode_3_y",
                        "ClmProcedureCode_4_y",
                        "ClmProcedureCode_5_y",
                        "ClmProcedureCode_6_y",
                        "AdmissionDt",
                        "DischargeDt",
                        "DeductibleAmtPaid_y",
                        "DiagnosisGroupCode",
                    ],
                    axis=1,
                )

                df_t[
                    [
                        "ClmDiagnosisCode_1_x",
                        "ClmDiagnosisCode_2_x",
                        "ClmDiagnosisCode_3_x",
                        "ClmDiagnosisCode_4_x",
                        "ClmDiagnosisCode_5_x",
                        "ClmDiagnosisCode_6_x",
                        "ClmDiagnosisCode_7_x",
                        "ClmDiagnosisCode_8_x",
                        "ClmDiagnosisCode_9_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                ] = df_t[
                    [
                        "ClmDiagnosisCode_1_x",
                        "ClmDiagnosisCode_2_x",
                        "ClmDiagnosisCode_3_x",
                        "ClmDiagnosisCode_4_x",
                        "ClmDiagnosisCode_5_x",
                        "ClmDiagnosisCode_6_x",
                        "ClmDiagnosisCode_7_x",
                        "ClmDiagnosisCode_8_x",
                        "ClmDiagnosisCode_9_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                ].replace(
                    np.nan, 0
                )
                df_t[
                    [
                        "ClmProcedureCode_1_x",
                        "ClmProcedureCode_2_x",
                        "ClmProcedureCode_3_x",
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ]
                ] = df_t[
                    [
                        "ClmProcedureCode_1_x",
                        "ClmProcedureCode_2_x",
                        "ClmProcedureCode_3_x",
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ]
                ].replace(
                    np.nan, 0
                )

                df_t["ClmAdmitDiagnosisCode_x"] = df_t[
                    "ClmAdmitDiagnosisCode_x"
                ].replace(np.nan, 0)
                df_t[
                    ["AttendingPhysician_x", "OperatingPhysician_x", "OtherPhysician_x"]
                ] = df_t[
                    ["AttendingPhysician_x", "OperatingPhysician_x", "OtherPhysician_x"]
                ].replace(
                    np.nan, "NA"
                )

                df_t["DOB"] = pd.to_datetime(df_t["DOB"], format="%Y-%m-%d")
                df_t["DOD"] = pd.to_datetime(df_t["DOD"], format="%Y-%m-%d")
                df_t["DOD"].fillna(df_t["DOD"].max(), inplace=True)
                df_t["ClaimStartDt_x"] = pd.to_datetime(
                    df_t["ClaimStartDt_x"], format="%Y-%m-%d"
                )
                df_t["ClaimEndDt_x"] = pd.to_datetime(
                    df_t["ClaimEndDt_x"], format="%Y-%m-%d"
                )
                df_t["Claim_Duration"] = (
                    df_t["ClaimEndDt_x"] - df_t["ClaimStartDt_x"]
                ).dt.days
                df_t["B_Age"] = round(((df_t["DOD"] - df_t["DOB"]).dt.days) / 365, 1)

                # df_t is the dataframe, a='ClaimID',x= Attending Physician, y=Operating Physician, z=Other Physician

                def AllPhy_totClaims(df_t, a, x, y, z):
                    df_t["AttPhy_TC"] = df_t.groupby(x)[a].transform("count")
                    df_t["OptPhy_TC"] = df_t.groupby(y)[a].transform("count")
                    df_t["OthPhy_TC"] = df_t.groupby(z)[a].transform("count")
                    df_t["AttPhy_TC"].fillna(0, inplace=True)
                    df_t["OptPhy_TC"].fillna(0, inplace=True)
                    df_t["OthPhy_TC"].fillna(0, inplace=True)
                    df_t["AllPhy_totClaims"] = (
                        df_t["AttPhy_TC"] + df_t["OptPhy_TC"] + df_t["OthPhy_TC"]
                    )
                    df_t.drop(
                        ["AttPhy_TC", "OptPhy_TC", "OthPhy_TC"], axis=1, inplace=True
                    )
                    return df_t

                AllPhy_totClaims(
                    df_t,
                    "ClaimID",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                )

                # Provider and all physicians interaction

                # df_t is dataframe, a=Provider, x=Attending Physician, y=Operating Physician, z= Other Physician

                def Prvd_AllPhy(df_t, a, x, y, z):
                    df_t["Prvd_AttPhy"] = df_t.groupby(a)[x].transform("count")
                    df_t["Prvd_OptPhy"] = df_t.groupby(a)[y].transform("count")
                    df_t["Prvd_OthPhy"] = df_t.groupby(a)[z].transform("count")
                    df_t["Prvd_AllPhy"] = (
                        df_t["Prvd_AttPhy"] + df_t["Prvd_OptPhy"] + df_t["Prvd_OthPhy"]
                    )
                    df_t.drop(
                        ["Prvd_AttPhy", "Prvd_OptPhy", "Prvd_OthPhy"],
                        axis=1,
                        inplace=True,
                    )
                    return df_t

                Prvd_AllPhy(
                    df_t,
                    "Provider",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                )

                # Unique claim diagnosis used by providers

                # df_t is dataframe, a=provider, x= claimadmitdiagnosiscode,

                def Pvrd_CADC(df_t, a, x):
                    df_t["Pvrd_CADC"] = df_t.groupby(a)[x].transform("nunique")
                    return df_t

                Pvrd_CADC(df_t, "Provider", "ClmAdmitDiagnosisCode_x")

                # Sum of Insc Claim Re-Imb Amount for a Provider

                # df_t is dataframe, a= provider,x=Insurance claim amt reinmbursed

                def Pvrd_InsClnReimAmt(df_t, a, x):
                    df_t["Pvrd_InsClnReimAmt"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_InsClnReimAmt(df_t, "Provider", "InscClaimAmtReimbursed_x")

                df_t["RenalDiseaseIndicator"].replace({0: 0, "Y": 1}, inplace=True)
                df_t["RenalDiseaseIndicator"] = df_t["RenalDiseaseIndicator"].astype(
                    "int"
                )

                # renal disease indicator seen by provider
                # df_t is dataframe, a= provider,x=renal disease indicator
                def Pvrd_RDI(df_t, a, x):
                    df_t["Pvrd_RDI"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_RDI(df_t, "Provider", "RenalDiseaseIndicator")

                # df_t is dataframe, a= provider,x=deductible amount paid
                def Pvrd_DedAmtPaid(df_t, a, x):
                    df_t["Pvrd_DedAmtPaid"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_DedAmtPaid(df_t, "Provider", "DeductibleAmtPaid_x")

                # df_t is dataframe, a= provider,x= IP annual amt reimbursement
                def Pvrd_IPAnAmtReim(df_t, a, x):
                    df_t["Pvrd_IPAnAmtReim"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_IPAnAmtReim(df_t, "Provider", "IPAnnualReimbursementAmt")

                # df_t is dataframe, a= provider,x= OP annual amt reimbursement
                def Pvrd_OPAnAmtReim(df_t, a, x):
                    df_t["Pvrd_OPAnAmtReim"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_OPAnAmtReim(df_t, "Provider", "OPAnnualReimbursementAmt")

                # df_t is dataframe, a= provider,x= OP annual deductible amt
                def Pvrd_OPAnDedAmt(df_t, a, x):
                    df_t["Pvrd_OPAnDedAmt"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_OPAnDedAmt(df_t, "Provider", "OPAnnualDeductibleAmt")

                # df_t is dataframe, a= provider,x= IP annual deductible amt
                def Pvrd_IPAnDedAmt(df_t, a, x):
                    df_t["Pvrd_IPAnDedAmt"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_IPAnDedAmt(df_t, "Provider", "IPAnnualDeductibleAmt")

                # df_t is dataframe, a= provider,x=Claim duration
                def Pvrd_ClmDur(df_t, a, x):
                    df_t["Pvrd_ClmDur"] = df_t.groupby(a)[x].transform("sum")
                    return df_t

                Pvrd_ClmDur(df_t, "Provider", "Claim_Duration")

                def create_agg_feats(df_t, grp_col, feat_name, operation="sum"):
                    feat_1 = feat_name + "_Insc_ReImb_Amt"
                    df_t[feat_1] = df_t.groupby(grp_col)[
                        "InscClaimAmtReimbursed_x"
                    ].transform(operation)

                    feat_2 = feat_name + "_DedAmtPaid"
                    df_t[feat_2] = df_t.groupby(grp_col)[
                        "DeductibleAmtPaid_x"
                    ].transform(operation)

                    feat_3 = feat_name + "_IP_Annual_ReImb_Amt"
                    df_t[feat_3] = df_t.groupby(grp_col)[
                        "IPAnnualReimbursementAmt"
                    ].transform(operation)

                    feat_4 = feat_name + "_IP_Annual_Ded_Amt"
                    df_t[feat_4] = df_t.groupby(grp_col)[
                        "IPAnnualDeductibleAmt"
                    ].transform(operation)

                    feat_5 = feat_name + "_OP_Annual_ReImb_Amt"
                    df_t[feat_5] = df_t.groupby(grp_col)[
                        "OPAnnualReimbursementAmt"
                    ].transform(operation)

                    feat_6 = feat_name + "_OP_Annual_Ded_Amt"
                    df_t[feat_6] = df_t.groupby(grp_col)[
                        "OPAnnualDeductibleAmt"
                    ].transform(operation)

                    feat_7 = feat_name + "_Claim_Duration"
                    df_t[feat_7] = df_t.groupby(grp_col)["Claim_Duration"].transform(
                        operation
                    )

                create_agg_feats(df_t, grp_col="BeneID", feat_name="BENE")
                create_agg_feats(
                    df_t, grp_col="AttendingPhysician_x", feat_name="ATT_PHY"
                )
                create_agg_feats(
                    df_t, grp_col="OperatingPhysician_x", feat_name="OPT_PHY"
                )
                create_agg_feats(df_t, grp_col="OtherPhysician_x", feat_name="OTH_PHY")
                create_agg_feats(
                    df_t,
                    grp_col="ClmAdmitDiagnosisCode_x",
                    feat_name="Claim_Admit_Diag_Code",
                )

                df_t.drop(
                    [
                        "ClmProcedureCode_4_x",
                        "ClmProcedureCode_5_x",
                        "ClmProcedureCode_6_x",
                    ],
                    axis=1,
                    inplace=True,
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_1_x", feat_name="Claim_DiagCode1"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_2_x", feat_name="Claim_DiagCode2"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_3_x", feat_name="Claim_DiagCode3"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_4_x", feat_name="Claim_DiagCode4"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_5_x", feat_name="Claim_DiagCode5"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_6_x", feat_name="Claim_DiagCode6"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_7_x", feat_name="Claim_DiagCode7"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_8_x", feat_name="Claim_DiagCode8"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_9_x", feat_name="Claim_DiagCode9"
                )
                create_agg_feats(
                    df_t, grp_col="ClmDiagnosisCode_10_x", feat_name="Claim_DiagCode10"
                )

                create_agg_feats(
                    df_t, grp_col="ClmProcedureCode_1_x", feat_name="Claim_ProcCode1"
                )
                create_agg_feats(
                    df_t, grp_col="ClmProcedureCode_2_x", feat_name="Claim_ProcCode2"
                )
                create_agg_feats(
                    df_t, grp_col="ClmProcedureCode_3_x", feat_name="Claim_ProcCode3"
                )

                # PROVIDER <--> other features :: To get claim counts

                df_t["ClmCount_Provider"] = df_t.groupby(["Provider"])[
                    "ClaimID"
                ].transform("count")
                df_t["ClmCount_Provider_BeneID"] = df_t.groupby(["Provider", "BeneID"])[
                    "ClaimID"
                ].transform("count")
                df_t["ClmCount_Provider_AttendingPhysician"] = df_t.groupby(
                    ["Provider", "AttendingPhysician_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_OtherPhysician"] = df_t.groupby(
                    ["Provider", "OtherPhysician_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_OperatingPhysician"] = df_t.groupby(
                    ["Provider", "OperatingPhysician_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmAdmitDiagnosisCode"] = df_t.groupby(
                    ["Provider", "ClmAdmitDiagnosisCode_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmProcedureCode_1"] = df_t.groupby(
                    ["Provider", "ClmProcedureCode_1_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmProcedureCode_2"] = df_t.groupby(
                    ["Provider", "ClmProcedureCode_2_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmProcedureCode_3"] = df_t.groupby(
                    ["Provider", "ClmProcedureCode_3_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_1"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_1_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_2"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_2_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_3"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_3_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_4"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_4_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_5"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_5_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_6"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_6_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_7"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_7_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_8"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_8_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_9"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_9_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_ClmDiagnosisCode_10"] = df_t.groupby(
                    ["Provider", "ClmDiagnosisCode_10_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> ATTENDING PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OPERATING PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OTHER PHYSICIAN <--> PROCEDURE CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_1"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_1_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_2"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_2_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmProcedureCode_3"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmProcedureCode_3_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> ATTENDING PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_4"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_4_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_5"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_5_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_6"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_6_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_7"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_7_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_8"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_8_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_9"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_9_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_10"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "AttendingPhysician_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OPERATING PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_4"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_4_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_5"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_5_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_6"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_6_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_7"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_7_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_8"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_8_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_9"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_9_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OperatingPhysician_ClmDiagnosisCode_10"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "OperatingPhysician_x",
                        "ClmDiagnosisCode_10_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> OTHER PHYSICIAN <--> DIAGNOSIS CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_1"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_1_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_2"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_2_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_3"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_3_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_4"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_4_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_5"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_5_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_6"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_6_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_7"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_7_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_8"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_8_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_9"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_9_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_OtherPhysician_ClmDiagnosisCode_10"
                ] = df_t.groupby(
                    ["Provider", "BeneID", "OtherPhysician_x", "ClmDiagnosisCode_10_x"]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                # PROVIDER <--> BENE <--> PROCEDURE CODES :: To get claim counts
                df_t["ClmCount_Provider_BeneID_ClmProcedureCode_1"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_1_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmProcedureCode_2"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_2_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmProcedureCode_3"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmProcedureCode_3_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> DIAGNOSIS CODES :: To get claim counts
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_1_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_2"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_2_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_3"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_3_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_4"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_4_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_5"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_5_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_6"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_6_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_7"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_7_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_8"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_8_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_9"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_9_x"]
                )["ClaimID"].transform("count")
                df_t["ClmCount_Provider_BeneID_ClmDiagnosisCode_10"] = df_t.groupby(
                    ["Provider", "BeneID", "ClmDiagnosisCode_10_x"]
                )["ClaimID"].transform("count")

                # PROVIDER <--> BENE <--> DIAGNOSIS CODES <--> PROCEDURE CODES :: To get claim counts
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_1_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_2_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_2_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_3_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_3_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_4_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_4_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_5_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_5_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_6_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_6_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_7_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_7_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_8_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_8_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_9_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_9_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_1"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_1_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_2"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_2_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )
                df_t[
                    "ClmCount_Provider_BeneID_ClmDiagnosisCode_10_ClmProcedureCode_3"
                ] = df_t.groupby(
                    [
                        "Provider",
                        "BeneID",
                        "ClmDiagnosisCode_10_x",
                        "ClmProcedureCode_3_x",
                    ]
                )[
                    "ClaimID"
                ].transform(
                    "count"
                )

                columns_to_remove = [
                    "BeneID",
                    "ClaimID",
                    "ClaimStartDt_x",
                    "ClaimEndDt_x",
                    "AttendingPhysician_x",
                    "OperatingPhysician_x",
                    "OtherPhysician_x",
                    "ClmAdmitDiagnosisCode_x",
                    "ClmDiagnosisCode_1_x",
                    "ClmDiagnosisCode_2_x",
                    "ClmDiagnosisCode_3_x",
                    "ClmDiagnosisCode_4_x",
                    "ClmDiagnosisCode_5_x",
                    "ClmDiagnosisCode_6_x",
                    "ClmDiagnosisCode_7_x",
                    "ClmDiagnosisCode_8_x",
                    "ClmDiagnosisCode_9_x",
                    "ClmDiagnosisCode_10_x",
                    "ClmProcedureCode_1_x",
                    "ClmProcedureCode_2_x",
                    "ClmProcedureCode_3_x",
                    "DOB",
                    "DOD",
                    "State",
                    "County",
                ]
                df_t.drop(columns_to_remove, axis=1, inplace=True)

                df_t["Gender"].replace({2.0: 0, 1.0: 1}, inplace=True)
                df_t["ChronicCond_Alzheimer"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_Heartfailure"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_KidneyDisease"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_Cancer"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_ObstrPulmonary"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_Depression"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_Diabetes"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_IschemicHeart"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_Osteoporasis"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_rheumatoidarthritis"].replace(2.0, 0, inplace=True)
                df_t["ChronicCond_stroke"].replace(2.0, 0, inplace=True)

                df_t = pd.get_dummies(df_t, columns=["Race"])
                df_t = df_t.groupby(["Provider"], as_index=False).agg("sum")

                X = df_t.iloc[:, 1:]
                X_t = stdscl.fit_transform(X)

                fraud_prob = xclf.predict_proba(X_t)
                df_t["Fraud_Probability"] = fraud_prob[:, 1]

                def Fraud_Potential(a):
                    if a < 0.1:
                        return "Negligible"
                    elif (a > 0.1) & (a < 0.4):
                        return "Low Potential"
                    elif (a > 0.4) & (a < 0.6):
                        return "Mediocre Potential"
                    elif (a > 0.6) & (a < 0.8):
                        return "High Potential"
                    else:
                        return "Very High Potential"

                df_t["Fraud_Potential"] = df_t["Fraud_Probability"].apply(
                    Fraud_Potential
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.write("Very High Fraud Potential")
                c1.dataframe(
                    df_t[df_t["Fraud_Potential"] == "Very High Potential"]["Provider"]
                )
                c2.write("High Fraud Potential")
                c2.dataframe(
                    df_t[df_t["Fraud_Potential"] == "High Potential"]["Provider"]
                )
                c3.write("Mediocre Fraud Potential")
                c3.dataframe(
                    df_t[df_t["Fraud_Potential"] == "Mediocre Potential"]["Provider"]
                )
                c4.write("Low Fraud Potential")
                c4.dataframe(
                    df_t[df_t["Fraud_Potential"] == "Low Potential"]["Provider"]
                )
                c5.write("Negligible Fraud Potential")
                c5.dataframe(df_t[df_t["Fraud_Potential"] == "Negligible"]["Provider"])
                
                
                