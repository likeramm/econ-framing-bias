"""Streamlit 대시보드"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.set_page_config(
        page_title="경제뉴스 프레이밍 편향 탐지기",
        page_icon="📊",
        layout="wide",
    )

    st.title("경제뉴스 프레이밍 편향 탐지기 + 주가 영향 분석")

    # 사이드바
    st.sidebar.header("분석 설정")

    # TODO: 대시보드 구현
    # - 언론사별 편향 프로파일
    # - 이벤트 타임라인 + 주가 오버레이
    # - 실시간 기사 분석 데모

    st.info("대시보드 개발 중입니다.")


if __name__ == "__main__":
    main()
