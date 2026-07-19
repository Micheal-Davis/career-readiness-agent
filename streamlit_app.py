"""Local Streamlit workspace backed exclusively by the Career Agent API."""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("CAREER_AGENT_API_URL", "http://127.0.0.1:8000")
REQUIRED_FIELDS = {
    "project": ("technologies", "contribution", "result"),
    "work": ("responsibilities", "work_content"),
    "competition": ("outcome", "contribution"),
    "course": ("course_or_activity", "outcome", "related_work"),
    "campus": ("responsibilities", "developed_capabilities"),
}
FIELD_LABELS = {
    "technologies": "技术/工具", "contribution": "个人贡献", "result": "结果",
    "responsibilities": "职责", "work_content": "工作内容",
    "outcome": "结果/奖项", "course_or_activity": "课程或活动",
    "related_work": "相关实践", "developed_capabilities": "锻炼的能力",
}
EVIDENCE_TYPE_LABELS = {
    "project": "项目经历",
    "work": "实习/工作经历",
    "competition": "竞赛经历",
    "course": "课程与实践",
    "campus": "校园经历",
}


def api_request(
    method: str, api_url: str, path: str, *, json: dict[str, Any] | None = None, files: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[Any | None, str | None]:
    try:
        response = requests.request(method, f"{api_url.rstrip('/')}{path}", json=json, files=files, params=params, timeout=90)
    except requests.RequestException:
        return None, "无法连接后端，请先启动 FastAPI 服务并检查后端地址。"
    try:
        data = response.json() if response.content else None
    except ValueError:
        return None, "后端返回了无法识别的响应。"
    if not response.ok:
        return None, (data or {}).get("detail", "后端暂时不可用，请稍后重试。")
    return data, None


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"来源（{len(sources)}）"):
        for source in sources:
            st.markdown(f"**{source['source']}**  \n路径：`{source['source_path']}`  \n切片：{source['chunk_index']}")


def initialize_state() -> None:
    defaults = {
        "api_url": DEFAULT_API_URL,
        "thread_id": "",
        "thread_id_input": "",
        "history": [],
        "profile_id": "",
        "workspace_mode": "home",
        "agent_step": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if pending_thread_id := st.session_state.pop("pending_thread_id", None):
        st.session_state.thread_id = pending_thread_id
        st.session_state.thread_id_input = pending_thread_id


def sync_thread_id() -> None:
    st.session_state.thread_id = st.session_state.thread_id_input.strip()


def start_new_conversation() -> None:
    st.session_state.thread_id = ""
    st.session_state.thread_id_input = ""
    st.session_state.history = []


AGENT_STEPS = ("隐私设置", "能力档案", "岗位机会", "准备任务", "求职建议")


def enter_workspace(mode: str) -> None:
    st.session_state.workspace_mode = mode


def set_agent_step(step: int) -> None:
    st.session_state.agent_step = step


def render_home_page() -> None:
    st.title("Career Knowledge Agent")
    st.caption("选择你现在要做的事。普通聊天与求职准备是两条独立路径。")
    chat, agent = st.columns(2)
    with chat:
        st.subheader("普通聊天")
        st.write("围绕已建立的本地知识库自由提问，不进入求职准备流程。")
        st.button("进入普通聊天", use_container_width=True, on_click=enter_workspace, args=("chat",))
    with agent:
        st.subheader("开始求职 Agent")
        st.write("按步骤建立能力档案、分析岗位，并得到可追溯的求职建议。")
        st.button("开始求职准备", type="primary", use_container_width=True, on_click=enter_workspace, args=("agent",))


def render_agent_header() -> None:
    current_step = st.session_state.agent_step
    st.caption(f"求职 Agent · 第 {current_step + 1} / {len(AGENT_STEPS)} 步")
    selected = st.radio(
        "准备进度（可回到任意步骤修改）",
        options=range(len(AGENT_STEPS)),
        index=current_step,
        horizontal=True,
        format_func=lambda step: f"{step + 1}. {AGENT_STEPS[step]}",
    )
    if selected != current_step:
        set_agent_step(selected)
        st.rerun()


def render_agent_step_controls() -> None:
    step = st.session_state.agent_step
    previous, spacer, next_step = st.columns([1, 3, 1])
    if step > 0 and previous.button("← 上一步", use_container_width=True):
        set_agent_step(step - 1)
        st.rerun()
    if step < len(AGENT_STEPS) - 1 and next_step.button("下一步 →", use_container_width=True):
        if step == 1 and not st.session_state.profile_id:
            st.warning("请先上传简历，或选择“从空白档案开始”。之后仍可随时返回上传新简历。")
        else:
            set_agent_step(step + 1)
            st.rerun()


def render_consent_page(api_url: str) -> None:
    st.title("开始使用")
    st.caption("同意可分别撤回。联网研究只发送搜索关键词，不会自动上传简历或能力档案。")
    current, error = api_request("GET", api_url, "/consents")
    if error:
        st.error(error)
        return
    labels = {
        "local_processing": "本地处理：在本机保存档案、文件和会话",
        "model_analysis": "模型分析：向配置的模型服务发送聊天内容及上传简历的文本，用于结构化提取",
        "web_research": "联网研究：向搜索服务发送公开研究关键词",
    }
    for scope, label in labels.items():
        enabled = st.toggle(label, value=current.get(scope, False), key=f"consent_{scope}")
        if enabled != current.get(scope, False):
            _, update_error = api_request("PUT", api_url, f"/consents/{scope}", json={"granted": enabled})
            if update_error:
                st.error(update_error)
            else:
                st.rerun()
    st.info("聊天功能需要先开启“模型分析”。")


def render_profile_page(api_url: str) -> None:
    st.title("能力档案")
    st.caption("上传简历后只生成待确认草稿；原始文件不会被覆盖。")
    if not st.session_state.profile_id:
        uploaded = st.file_uploader("上传已有简历（可选：DOCX / PDF / TXT / MD）", type=["docx", "pdf", "txt", "md"])
        left, right = st.columns(2)
        if left.button("从空白档案开始", use_container_width=True):
            result, error = api_request("POST", api_url, "/profiles/empty")
            if error:
                st.error(error)
            else:
                st.session_state.profile_id = result["profile_id"]
                st.rerun()
        if right.button("上传并提取草稿", disabled=uploaded is None, use_container_width=True):
            result, error = api_request(
                "POST", api_url, "/profiles/resume",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
            )
            if error:
                st.error(error)
            else:
                st.session_state.profile_id = result["profile_id"]
                st.rerun()
        return

    st.success(f"当前能力档案：`{st.session_state.profile_id}`")
    if st.button("上传新简历或重新选择档案", use_container_width=True):
        # Old profiles remain intact locally; this only switches the active profile.
        st.session_state.profile_id = ""
        st.session_state.pop("resume_suggestions", None)
        st.rerun()
    drafts, error = api_request("GET", api_url, f"/profiles/{st.session_state.profile_id}/evidence/drafts")
    if error:
        st.error(error)
        return
    confirmed, confirmed_error = api_request("GET", api_url, f"/profiles/{st.session_state.profile_id}/evidence")
    if confirmed_error:
        st.error(confirmed_error)
        return

    st.subheader("经历栏目")
    for evidence_type in REQUIRED_FIELDS:
        _render_evidence_category(
            api_url,
            st.session_state.profile_id,
            evidence_type,
            [item for item in drafts if item["evidence_type"] == evidence_type],
            [item for item in confirmed if item["evidence_type"] == evidence_type],
        )


def _render_evidence_category(
    api_url: str,
    profile_id: str,
    evidence_type: str,
    drafts: list[dict[str, Any]],
    confirmed: list[dict[str, Any]],
) -> None:
    """Keep creation, review, and deletion of one evidence type in one visible category."""
    with st.container(border=True):
        st.subheader(f"{EVIDENCE_TYPE_LABELS[evidence_type]}（{len(drafts) + len(confirmed)}）")
        with st.form(f"new_evidence_{evidence_type}"):
            title = st.text_input("标题", key=f"new_title_{evidence_type}")
            details = {
                field: st.text_area(FIELD_LABELS[field], key=f"new_{evidence_type}_{field}")
                for field in REQUIRED_FIELDS[evidence_type]
            }
            add = st.form_submit_button(f"添加{EVIDENCE_TYPE_LABELS[evidence_type]}")
        if add:
            _, error = api_request(
                "POST", api_url, f"/profiles/{profile_id}/evidence",
                json={"title": title, "evidence_type": evidence_type, "details": details},
            )
            if error:
                st.error(error)
            else:
                st.rerun()

        for draft in drafts:
            with st.expander(f"待确认：{draft['title']}", expanded=True):
                if draft["details"].get("extracted_text"):
                    st.caption("从原文件提取的原始文本")
                    st.code(draft["details"]["extracted_text"])
                with st.form(f"draft_{draft['id']}"):
                    title = st.text_input("标题", value=draft["title"])
                    details = {
                        field: st.text_area(FIELD_LABELS[field], value=draft["details"].get(field, ""))
                        for field in REQUIRED_FIELDS[evidence_type]
                    }
                    save = st.form_submit_button("保存并确认")
                if save:
                    _, error = api_request(
                        "PUT", api_url, f"/evidence/{draft['id']}",
                        json={"title": title, "evidence_type": evidence_type, "details": details},
                    )
                    if error:
                        st.error(error)
                    else:
                        _, error = api_request("POST", api_url, f"/evidence/{draft['id']}/confirm")
                        if error:
                            st.error(error)
                        else:
                            st.rerun()
                if st.button("删除这条经历", key=f"delete_draft_{draft['id']}"):
                    _, error = api_request("DELETE", api_url, f"/evidence/{draft['id']}")
                    if error:
                        st.error(error)
                    else:
                        st.rerun()

        for record in confirmed:
            with st.expander(f"已确认：{record['title']}"):
                for field, value in record["details"].items():
                    if field != "extracted_text" and value:
                        st.markdown(f"- **{FIELD_LABELS.get(field, field)}**：{value}")
                if st.button("删除这条经历", key=f"delete_evidence_{record['id']}"):
                    _, error = api_request("DELETE", api_url, f"/evidence/{record['id']}")
                    if error:
                        st.error(error)
                    else:
                        st.rerun()


def render_jobs_page(api_url: str) -> None:
    st.title("岗位机会")
    st.caption("粘贴一个或多个岗位 JD。系统先拆分为待确认岗位，不会把多个方向混为一个目标。")
    with st.form("jd_import"):
        jd_text = st.text_area("岗位 JD", height=220, placeholder="招聘方向：服务端工程师\n岗位职责：…")
        import_jd = st.form_submit_button("拆分为岗位草稿", use_container_width=True)
    if import_jd:
        _, error = api_request("POST", api_url, "/jobs/jd", json={"jd_text": jd_text})
        if error:
            st.error(error)
        else:
            st.success("已创建岗位草稿，请在下方确认。")
            st.rerun()
    jobs, error = api_request("GET", api_url, "/jobs")
    if error:
        st.error(error)
        return
    if not jobs:
        st.info("还没有岗位机会。")
        return
    confirmed_jobs = [job for job in jobs if job["confirmation_status"] == "confirmed"]
    if confirmed_jobs:
        active, active_error = api_request("GET", api_url, "/planning/active-opportunities")
        if active_error:
            st.error(active_error)
        else:
            labels = {job["id"]: job["title"] for job in confirmed_jobs}
            selected_ids = st.multiselect(
                "当前活跃岗位（最多 3 个）",
                options=list(labels),
                default=[item for item in active["opportunity_ids"] if item in labels],
                format_func=labels.get,
            )
            if st.button("保存活跃岗位", use_container_width=True):
                _, active_update_error = api_request(
                    "PUT", api_url, "/planning/active-opportunities", json={"opportunity_ids": selected_ids}
                )
                if active_update_error:
                    st.error(active_update_error)
                else:
                    st.success("活跃岗位已保存。")
                    st.rerun()
            if active["opportunity_ids"]:
                primary_id = st.selectbox(
                    "本周主岗位（可选）", options=[""] + active["opportunity_ids"],
                    index=([""] + active["opportunity_ids"]).index(active["primary_opportunity_id"] or ""),
                    format_func=lambda item: "暂不选择" if not item else labels[item],
                )
                if primary_id and st.button("保存主岗位", use_container_width=True):
                    _, primary_error = api_request(
                        "PUT", api_url, "/planning/primary-opportunity",
                        json={"opportunity_id": primary_id},
                    )
                    if primary_error:
                        st.error(primary_error)
                    else:
                        st.success("主岗位已保存。")
                        st.rerun()
    for job in jobs:
        status = "已确认" if job["confirmation_status"] == "confirmed" else "待确认"
        with st.expander(f"{status}｜{job['title']}", expanded=job["confirmation_status"] == "draft"):
            st.code(job["source_jd_text"])
            if job["confirmation_status"] == "draft":
                title = st.text_input("岗位名称", value=job["title"], key=f"job_title_{job['id']}")
                left, right = st.columns(2)
                if left.button("保存名称", key=f"save_job_{job['id']}", use_container_width=True):
                    _, update_error = api_request("PUT", api_url, f"/jobs/{job['id']}", json={"title": title})
                    if update_error:
                        st.error(update_error)
                    else:
                        st.rerun()
                if right.button("确认岗位", key=f"confirm_job_{job['id']}", use_container_width=True):
                    _, confirm_error = api_request("POST", api_url, f"/jobs/{job['id']}/confirm")
                    if confirm_error:
                        st.error(confirm_error)
                    else:
                        if st.session_state.profile_id:
                            _, analysis_error = api_request(
                                "POST", api_url, f"/jobs/{job['id']}/analyze",
                                json={"profile_id": st.session_state.profile_id},
                            )
                            if analysis_error:
                                st.warning(f"岗位已确认，但自动分析暂未完成：{analysis_error}")
                            else:
                                st.success("岗位已确认，AI 已自动生成岗位要求矩阵。")
                        else:
                            st.info("岗位已确认；创建能力档案后会自动分析岗位要求。")
                        st.rerun()
            else:
                if st.session_state.profile_id and st.button("重新分析岗位要求", key=f"analyze_job_{job['id']}", use_container_width=True):
                    _, analysis_error = api_request(
                        "POST", api_url, f"/jobs/{job['id']}/analyze",
                        json={"profile_id": st.session_state.profile_id},
                    )
                    if analysis_error:
                        st.error(analysis_error)
                    else:
                        st.success("岗位分析已完成，要求矩阵已更新。")
                        st.rerun()
                elif not st.session_state.profile_id:
                    st.caption("先创建并确认能力档案后，才能进行个人匹配分析。")
                with st.expander("联网岗位研究"):
                    domains = st.text_input("企业官方域名（可选，逗号分隔）", key=f"research_domains_{job['id']}")
                    if st.button("研究公开岗位信息", key=f"research_job_{job['id']}"):
                        sources, research_error = api_request(
                            "POST", api_url, "/research",
                            json={"query": f"{job['title']} 招聘要求 团队技术", "first_party_domains": [item.strip() for item in domains.split(",") if item.strip()], "job_opportunity_id": job["id"]},
                        )
                        if research_error:
                            st.error(research_error)
                        elif not sources:
                            st.info("未找到可用公开来源。")
                        else:
                            for source in sources:
                                st.markdown(f"- `{source['tier']}` [{source['title']}]({source['url']})")
                saved_sources, saved_sources_error = api_request("GET", api_url, f"/jobs/{job['id']}/research-sources")
                if not saved_sources_error and saved_sources:
                    with st.expander(f"已保存的研究来源（{len(saved_sources)}）"):
                        for source in saved_sources:
                            st.markdown(f"- `{source['tier']}` [{source['title']}]({source['url']})")
                matrix, matrix_error = api_request("GET", api_url, f"/jobs/{job['id']}/requirements")
                if matrix_error:
                    st.error(matrix_error)
                    continue
                if matrix:
                    st.caption("岗位要求矩阵")
                    for requirement in matrix:
                        st.markdown(
                            f"- **{requirement['text']}** · {requirement['origin']} · "
                            f"{requirement['evidence_assessment']} · {requirement['priority']}"
                        )


def render_chat_page(api_url: str, *, include_profile_context: bool = False) -> None:
    st.title("普通聊天")
    st.caption("自由使用本地知识库聊天。开始聊天前，请先同意模型分析。")
    if st.session_state.thread_id:
        st.caption(f"当前会话：`{st.session_state.thread_id}`")
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                render_sources(item.get("sources", []))
    if prompt := st.chat_input("例如：介绍我的知识库项目"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("正在查询知识库..."):
                result, error = api_request("POST", api_url, "/chat", json={"message": prompt, "thread_id": st.session_state.thread_id or None, "profile_id": st.session_state.profile_id if include_profile_context else None})
            if error:
                st.error(error)
                return
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.session_state.history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            st.session_state.pending_thread_id = result["thread_id"]
            st.rerun()


def render_tasks_page(api_url: str) -> None:
    st.title("准备任务")
    st.caption("任务必须关联当前活跃岗位和具体要求；推荐理由会明确展示证据缺口、复用价值与时间可行性。")
    active, active_error = api_request("GET", api_url, "/planning/active-opportunities")
    if active_error:
        st.error(active_error)
        return
    if not active["opportunity_ids"]:
        st.info("请先在“岗位机会”页选择至少一个活跃岗位。")
        return
    requirements = []
    for job_id in active["opportunity_ids"]:
        matrix, matrix_error = api_request("GET", api_url, f"/jobs/{job_id}/requirements")
        if matrix_error:
            st.error(matrix_error)
            return
        requirements.extend(matrix)
    if not requirements:
        st.info("请先在活跃岗位中补充至少一项岗位要求。")
        return
    requirement_labels = {item["id"]: f"{item['text']}（{item['evidence_assessment']}）" for item in requirements}
    weekly_hours = st.number_input("本周可用于准备的小时数", min_value=1, max_value=80, value=8)
    with st.form("create_task"):
        title = st.text_input("任务标题", placeholder="完成 FastAPI 检索服务项目")
        description = st.text_area("可验证的任务说明", placeholder="实现接口、编写测试并提供可演示结果")
        requirement_ids = st.multiselect("关联的岗位要求", options=list(requirement_labels), format_func=requirement_labels.get)
        estimated_hours = st.number_input("预计耗时（小时）", min_value=1, max_value=200, value=4)
        create_task = st.form_submit_button("创建准备任务", use_container_width=True)
    if create_task:
        _, task_error = api_request(
            "POST", api_url, "/planning/tasks",
            json={"title": title, "description": description, "opportunity_ids": active["opportunity_ids"], "requirement_ids": requirement_ids, "estimated_hours": estimated_hours},
        )
        if task_error:
            st.error(task_error)
        else:
            st.success("任务已创建。")
            st.rerun()
    recommendation, recommendation_error = api_request("GET", api_url, "/planning/recommendation", params={"weekly_hours": int(weekly_hours)})
    if recommendation_error:
        st.error(recommendation_error)
    elif recommendation:
        st.subheader("建议下一步")
        st.markdown(f"**{recommendation['task']['title']}** · 预计 {recommendation['task']['estimated_hours']} 小时")
        for reason in recommendation["rationale"]:
            st.write(f"- {reason}")
        with st.expander("提交完成证据", expanded=False):
            with st.form(f"completion_{recommendation['task']['id']}"):
                evidence_title = st.text_input("项目标题")
                technologies = st.text_input("技术/工具")
                contribution = st.text_area("个人贡献")
                result = st.text_area("结果")
                work_link = st.text_input("作品链接（可选）")
                submit_completion = st.form_submit_button("提交完成证据", use_container_width=True)
            if submit_completion:
                submitted, completion_error = api_request(
                    "POST", api_url, f"/planning/tasks/{recommendation['task']['id']}/completion",
                    json={"project_evidence": {"title": evidence_title, "technologies": technologies, "contribution": contribution, "result": result}, "work_link": work_link or None},
                )
                if completion_error:
                    st.error(completion_error)
                else:
                    st.session_state.completion_task = submitted
                    st.rerun()
    else:
        st.info("创建至少一项任务后，系统会给出下一步推荐。")
    completion_task = st.session_state.get("completion_task")
    if completion_task and completion_task["status"] == "evidence_submitted":
        st.warning(f"“{completion_task['title']}”的完成证据已提交，请确认后才会标记为完成。")
        if st.button("确认任务完成", use_container_width=True):
            confirmed, confirm_error = api_request("POST", api_url, f"/planning/tasks/{completion_task['id']}/completion/confirm")
            if confirm_error:
                st.error(confirm_error)
            else:
                st.session_state.completion_task = confirmed
                st.success("任务已确认完成。")
    completion_task = st.session_state.get("completion_task")
    if completion_task and completion_task["status"] == "confirmed":
        if not st.session_state.profile_id:
            st.info("任务已完成。请先在“能力档案”页创建或导入一个档案，才能将成果入档。")
        elif st.button("将此成果加入能力档案", use_container_width=True):
            evidence, promotion_error = api_request(
                "POST", api_url, f"/planning/tasks/{completion_task['id']}/promote-evidence",
                json={"profile_id": st.session_state.profile_id},
            )
            if promotion_error:
                st.error(promotion_error)
            else:
                st.success(f"已将“{evidence['title']}”加入能力档案，并更新本地检索索引。")


def render_report_page(api_url: str) -> None:
    st.title("职业报告")
    st.caption("只汇总已确认的能力证据和你主动维护的岗位信息；不生成不可解释的匹配分数。")
    if not st.session_state.profile_id:
        st.info("请先在“能力档案”页创建或导入能力档案。")
        return
    weekly_hours = st.number_input("本周准备时间（小时）", min_value=1, max_value=80, value=8, key="report_hours")
    report, error = api_request("GET", api_url, "/reports/career", params={"profile_id": st.session_state.profile_id, "weekly_hours": int(weekly_hours)})
    if error:
        st.error(error)
        return
    st.metric("已确认能力证据", report["confirmed_evidence_count"])
    if not report["active_jobs"]:
        st.info("请先在“岗位机会”页选择活跃岗位。")
    for job in report["active_jobs"]:
        with st.expander(job["title"], expanded=True):
            if not job["requirements"]:
                st.caption("尚未建立岗位要求矩阵。")
            for item in job["requirements"]:
                st.markdown(f"- **{item['text']}** · `{item['evidence_assessment']}` · `{item['priority']}` · `{item['origin']}`")
    if report["next_task"]:
        task = report["next_task"]
        st.subheader("建议下一步")
        st.markdown(f"**{task['title']}** · 预计 {task['estimated_hours']} 小时")
        for reason in task["rationale"]:
            st.write(f"- {reason}")
    if report["active_jobs"]:
        job_labels = {item["id"]: item["title"] for item in report["active_jobs"]}
        target_job_id = st.selectbox("生成哪个岗位的简历建议", list(job_labels), format_func=job_labels.get)
        if st.button("生成证据约束的简历建议", use_container_width=True):
            optimization, optimization_error = api_request(
                "GET", api_url, "/resume/optimization",
                params={"profile_id": st.session_state.profile_id, "opportunity_id": target_job_id},
            )
            if optimization_error:
                st.error(optimization_error)
            elif not optimization["suggestions"]:
                st.info("没有足够完整的已确认经历可改写；请补充经历的个人贡献和结果。")
            else:
                st.session_state.resume_suggestions = optimization["suggestions"]
        for item in st.session_state.get("resume_suggestions", []):
            with st.expander(f"{item['section']}｜{item['evidence_title']}", expanded=True):
                st.text_area(
                    "可编辑建议", value=item["suggested_bullet"],
                    key=f"resume_suggestion_{item['evidence_id']}",
                )
                st.caption(f"依据：证据 `{item['evidence_id']}`；关联 {len(item['supporting_requirement_ids'])} 项岗位要求。")


def render_documents_page(api_url: str) -> None:
    st.title("资料库")
    st.caption("上传后的资料会自动建立索引；删除最后一份资料会安全清空活动索引，不会返回旧内容。")
    uploaded = st.file_uploader("上传资料", type=["txt", "md", "docx"])
    if st.button("上传并建立索引", disabled=uploaded is None, use_container_width=True):
        document, upload_error = api_request(
            "POST", api_url, "/documents",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
        )
        if upload_error:
            st.error(upload_error)
        elif document["index_status"] == "ready":
            st.success(f"已索引：{document['filename']}")
            st.rerun()
        else:
            st.warning(f"文件已保存，但索引失败：{document['last_error']}")
    documents, list_error = api_request("GET", api_url, "/documents")
    if list_error:
        st.error(list_error)
        return
    if not documents:
        st.info("资料库为空。")
        return
    for document in documents:
        left, right = st.columns([4, 1])
        left.markdown(f"**{document['filename']}**  \n索引状态：`{document['index_status']}`")
        if document["last_error"]:
            left.caption(document["last_error"])
        if right.button("删除", key=f"delete_document_{document['id']}", use_container_width=True):
            _, delete_error = api_request("DELETE", api_url, f"/documents/{document['id']}")
            if delete_error:
                st.error(delete_error)
            else:
                st.rerun()
        replacement = left.file_uploader("替换此资料", type=["txt", "md", "docx"], key=f"replace_document_{document['id']}")
        if replacement and left.button("替换并重建索引", key=f"replace_button_{document['id']}"):
            _, replace_error = api_request(
                "PUT", api_url, f"/documents/{document['id']}",
                files={"file": (replacement.name, replacement.getvalue(), replacement.type or "application/octet-stream")},
            )
            if replace_error:
                st.error(replace_error)
            else:
                st.success("资料已替换并重新建立索引。")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Career Knowledge Agent", page_icon="🎈")
    initialize_state()
    with st.sidebar:
        st.header("工作台")
        st.text_input("后端地址", key="api_url")
        if st.session_state.workspace_mode != "home":
            if st.button("← 返回首页", use_container_width=True):
                enter_workspace("home")
                st.rerun()
        if st.session_state.workspace_mode == "chat":
            st.text_input("会话 ID", key="thread_id_input", on_change=sync_thread_id)
            st.button("新建会话", use_container_width=True, on_click=start_new_conversation)
        if st.button("检查后端连接", use_container_width=True):
            health, error = api_request("GET", st.session_state.api_url, "/health")
            st.success("后端连接正常") if not error and health.get("status") == "ok" else st.error(error or "后端返回异常状态")
    if st.session_state.workspace_mode == "home":
        render_home_page()
        return
    if st.session_state.workspace_mode == "chat":
        render_chat_page(st.session_state.api_url)
        return

    render_agent_header()
    step = st.session_state.agent_step
    if step == 0:
        render_consent_page(st.session_state.api_url)
    elif step == 1:
        render_profile_page(st.session_state.api_url)
    elif step == 2:
        render_jobs_page(st.session_state.api_url)
    elif step == 3:
        render_tasks_page(st.session_state.api_url)
    else:
        render_report_page(st.session_state.api_url)
    render_agent_step_controls()

    if st.checkbox("管理资料库（可选）", key="show_documents"):
        render_documents_page(st.session_state.api_url)


if __name__ == "__main__":
    main()
