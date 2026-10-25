"""Gradio presentation; all diagnosis and approval operations go through HTTP."""

import os

# Set before importing Gradio; never send component data or analytics upstream.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import plotly.graph_objects as go

from pvdiag.ui.client import ApiClient, ApiError
from pvdiag.ui.controller import (
    Controller,
    chart_data,
    empty_session,
    peer_data,
    report_view,
    request_body,
)

METRICS = ["ac_power_kw", "dc_power_kw", "poa_wm2", "ambient_c", "module_c"]
CSS = """
.gradio-container {
  width: 100% !important; max-width: min(1440px, 100vw) !important;
  min-width: 0 !important; margin: auto; box-sizing: border-box;
}
.gradio-container input, .gradio-container textarea {font-variant-numeric: tabular-nums;}
#workspace {min-width: 0 !important;}
@media (max-width: 800px) {
  #workspace {flex-direction: column !important;}
  #workspace > div {min-width: 0 !important; width: 100% !important;}
  #workspace .row {flex-wrap: wrap !important;}
  #workspace button {min-width: 0 !important;}
}
"""


def line_figure(evidence, evidence_id, metric):
    frame = chart_data(evidence, evidence_id, metric)
    if frame.empty:
        return None
    unit = {
        "ac_power_kw": "kW",
        "dc_power_kw": "kW",
        "poa_wm2": "W/m²",
        "ambient_c": "°C",
        "module_c": "°C",
    }[metric]
    figure = go.Figure()
    for device, points in frame.groupby("设备", sort=False):
        figure.add_trace(
            go.Scatter(
                # Plotly date axes use these UTC calendar values without local conversion.
                x=points.UTC.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                y=points["数值"].tolist(),
                name=device,
                mode="lines+markers",
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>%{y}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=310,
        margin={"l": 60, "r": 15, "t": 15, "b": 55},
        xaxis={"type": "date", "title": "UTC", "tickformat": "%m-%d<br>%H:%M"},
        yaxis={"title": f"{metric} ({unit})"},
        legend={"orientation": "h", "y": -0.3},
    )
    return figure


def peer_figure(evidence, evidence_id):
    frame = peer_data(evidence, evidence_id)
    if frame.empty:
        return None
    figure = go.Figure(
        go.Bar(
            x=frame["设备"].tolist(),
            y=frame["低值占比"].tolist(),
            marker_color="#2563eb",
            hovertemplate="%{x}<br>日间低值占比 %{y:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=250,
        margin={"l": 60, "r": 15, "t": 15, "b": 45},
        xaxis={"title": "设备"},
        yaxis={"title": "日间低值占比", "type": "linear", "range": [0, 1], "tickformat": ".0%"},
    )
    return figure


def build_app(api_url="http://127.0.0.1:8000"):
    client = ApiClient(api_url)
    controller = Controller(client)
    with gr.Blocks(title="光伏诊断工作台", analytics_enabled=False) as demo:
        session = gr.State(empty_session())
        profiles = gr.State([])
        gr.Markdown("# 光伏诊断工作台\n选择数据范围，跟踪诊断证据，并核对待审批的核查工单。")
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=300):
                token = gr.Textbox(
                    label="API 访问令牌",
                    type="password",
                    max_length=1024,
                    info="使用当前身份的令牌连接；审批仍由服务端检查权限。",
                )
                connect = gr.Button("连接并读取配置")
                profile = gr.Dropdown(label="数据配置", choices=[], interactive=True)
                scope = gr.Textbox(label="可用范围", interactive=False, lines=3)
                devices = gr.Dropdown(label="诊断设备", choices=[], multiselect=True)
                start = gr.Textbox(
                    label="开始时间（UTC，含时区）", placeholder="2019-01-08T00:00:00Z"
                )
                end = gr.Textbox(
                    label="结束时间（UTC，不含此时刻）", placeholder="2019-01-15T00:00:00Z"
                )
                question = gr.Textbox(
                    label="诊断问题",
                    value="检查光伏逆变器的故障、时间范围及证据。",
                    lines=3,
                    max_length=2000,
                )
                begin = gr.Button("开始 / 重试诊断", variant="primary")
                reset = gr.Button("清空结果，准备新诊断")
                gr.Markdown(
                    "同一组输入的重试复用原运行。清空后再次开始会创建新运行；清空页面不会停止后台诊断。"
                )
                run_id = gr.Textbox(
                    label="运行 ID", placeholder="输入已有运行 ID，或开始后自动填写"
                )
                with gr.Row():
                    read = gr.Button("读取运行", size="sm")
                    reconnect = gr.Button("重连进度", size="sm")
                    resume = gr.Button("恢复运行", size="sm")
            with gr.Column(scale=2, min_width=400):
                status = gr.Textbox(label="运行状态", value="等待连接", interactive=False)
                notice = gr.Textbox(label="操作提示", interactive=False)
                with gr.Tab("诊断结果"):
                    summary = gr.Textbox(label="诊断摘要", interactive=False, lines=3)
                    faults = gr.Dataframe(
                        headers=["设备", "故障类型", "开始 UTC", "结束 UTC", "置信度", "依据"],
                        datatype="str",
                        interactive=False,
                        wrap=True,
                        max_height=320,
                    )
                    with gr.Accordion("完整结果与核验记录", open=False):
                        result_json = gr.JSON(label="诊断结果")
                with gr.Tab("证据图表"):
                    evidence_id = gr.Dropdown(label="证据", choices=[], interactive=True)
                    metric = gr.Dropdown(label="时序指标", choices=METRICS, value="poa_wm2")
                    line = gr.Plot(label="证据时序")
                    gr.Markdown(
                        "仅展示所选证据中工具返回的采样点；横轴为 UTC，不代表完整连续原始数据。"
                    )
                    bars = gr.Plot(label="同站对比")
                    gr.Markdown("peer_compare 返回的日间低值占比；选择该类证据后显示。")
                    evidence_json = gr.JSON(label="证据原文（含请求、单位和截断标记）")
                with gr.Tab("流式过程"):
                    progress = gr.Dataframe(
                        headers=["事件游标", "节点", "轮数", "工具调用数", "临时报告状态"],
                        interactive=False,
                        wrap=True,
                        max_height=450,
                    )
                with gr.Column(visible=False) as approval_panel:
                    gr.Markdown("### 人工审批\n核对以下完整草案。批准后仅创建本地核查工单。")
                    proposal_json = gr.JSON(label="待审批草案")
                    reason = gr.Textbox(label="审批说明（可选）", max_length=1000)
                    confirmed = gr.Checkbox(
                        label="我已核对上述草案，并确认记录本次人工决定", value=False
                    )
                    with gr.Row():
                        approve = gr.Button("批准并创建核查工单", interactive=False)
                        reject = gr.Button("拒绝此工单", interactive=False)

        outputs = [
            session,
            run_id,
            status,
            notice,
            summary,
            faults,
            result_json,
            evidence_id,
            metric,
            line,
            bars,
            evidence_json,
            progress,
            approval_panel,
            proposal_json,
            confirmed,
            reason,
            approve,
            reject,
        ]

        def render(state, message=""):
            view = report_view(state)
            evidence = view["evidence"]
            first = next(
                (k for k, e in evidence.items() if e["tool"] == "get_timeseries"),
                next(iter(evidence), None),
            )
            series = evidence.get(first, {}).get("result", {}).get("series", [])
            selected_metric = series[0].get("metric", "poa_wm2") if series else "poa_wm2"
            return {
                session: state,
                run_id: state["run_id"],
                status: view["status"],
                notice: message,
                summary: view["summary"],
                faults: view["faults"],
                result_json: view["result"],
                evidence_id: gr.update(
                    choices=[(f"{e['tool']} · {k}", k) for k, e in evidence.items()],
                    value=first,
                ),
                metric: selected_metric,
                line: line_figure(evidence, first, selected_metric),
                bars: peer_figure(evidence, first),
                evidence_json: evidence.get(first),
                progress: state["progress"],
                approval_panel: gr.update(visible=view["approval_enabled"]),
                proposal_json: view["proposal"],
                confirmed: False,
                reason: "",
                approve: gr.update(interactive=view["approval_enabled"]),
                reject: gr.update(interactive=view["approval_enabled"]),
            }

        async def drive(iterator, state):
            current = state
            try:
                async for current in iterator:
                    yield render(current)
            except ApiError as error:
                yield render(current, str(error))
            except Exception:  # noqa: BLE001 -- never expose headers or credential-bearing errors
                yield render(current, "界面请求未成功；请读取当前运行，或使用相同输入重试。")

        def select_profile(items, selected):
            item = next((p for p in items if p["profile_id"] == selected), None)
            if not item:
                return gr.update(choices=[], value=[]), "", "", ""
            text = (
                f"电站 {item['system_id']} · 模式 {item['mode']}\n"
                f"{item['start']} 至 {item['end']}\n"
                f"工单{'启用' if item.get('work_orders_enabled') else '关闭'}"
            )
            return (
                gr.update(choices=item["device_ids"], value=item["device_ids"]),
                item["start"],
                item["end"],
                text,
            )

        async def connect_profiles(access_token):
            try:
                items = await client.profiles(access_token)
                selected = items[0]["profile_id"] if items else None
                device_update, a, b, text = select_profile(items, selected)
                return (
                    items,
                    gr.update(choices=[p["profile_id"] for p in items], value=selected),
                    device_update,
                    a,
                    b,
                    text,
                    "配置已读取。",
                )
            except ApiError as error:
                return (
                    [],
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=[]),
                    "",
                    "",
                    "",
                    str(error),
                )

        async def start_run(access_token, state, items, selected, ids, a, b, prompt):
            try:
                item = next((p for p in items if p["profile_id"] == selected), None)
                body = request_body(item, ids, a, b, prompt)
            except ApiError as error:
                yield render(state, str(error))
                return
            async for update in drive(controller.start(access_token, state, body), state):
                yield update

        async def read_run(access_token, state, selected_run):
            async for update in drive(controller.read(access_token, state, selected_run), state):
                yield update

        async def reconnect_run(access_token, state):
            async for update in drive(controller.reconnect(access_token, state), state):
                yield update

        async def resume_run(access_token, state):
            async for update in drive(
                controller.reconnect(access_token, state, resume=True), state
            ):
                yield update

        async def approve_run(access_token, state, checked, text):
            async for update in drive(
                controller.decide(
                    access_token, state, approved=True, confirmed=checked, reason=text
                ),
                state,
            ):
                yield update

        async def reject_run(access_token, state, checked, text):
            async for update in drive(
                controller.decide(
                    access_token, state, approved=False, confirmed=checked, reason=text
                ),
                state,
            ):
                yield update

        def select_evidence(state, selected, selected_metric):
            evidence = report_view(state)["evidence"]
            return (
                evidence.get(selected),
                line_figure(evidence, selected, selected_metric),
                peer_figure(evidence, selected),
            )

        connect.click(
            connect_profiles,
            token,
            [profiles, profile, devices, start, end, scope, notice],
            api_visibility="private",
        )
        profile.input(
            select_profile,
            [profiles, profile],
            [devices, start, end, scope],
            api_visibility="private",
        )
        actions = [
            (begin, start_run, [token, session, profiles, profile, devices, start, end, question]),
            (read, read_run, [token, session, run_id]),
            (reconnect, reconnect_run, [token, session]),
            (resume, resume_run, [token, session]),
            (approve, approve_run, [token, session, confirmed, reason]),
            (reject, reject_run, [token, session, confirmed, reason]),
            (reset, lambda: render(empty_session()), []),
        ]
        for button, callback, inputs in actions:
            button.click(
                callback,
                inputs,
                outputs,
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id="run-actions",
                show_progress="minimal",
                trigger_mode="once",
            )
        gr.on(
            [evidence_id.input, metric.input],
            select_evidence,
            [session, evidence_id, metric],
            [evidence_json, line, bars],
            api_visibility="private",
        )
    return demo
