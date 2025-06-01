import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from shiny import App, ui, render, reactive
from dashscope import Application
import io
import base64
import json
from htmltools import HTML

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")
sns.set_style("whitegrid")

def load_data(level_name):
    file_map = {
        "一级产品": "company_data_一级产品.xlsx",
        "二级产品": "company_data_二级产品.xlsx",
        "三级产品": "company_data_三级产品.xlsx"
    }
    df = pd.read_excel(file_map[level_name], dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def load_cluster_data():
    """加载聚类数据"""
    try:
        # 尝试不同的文件路径
        possible_paths = [
            "cluster_data.xlsx",
            "./cluster_data.xlsx",
            "data/cluster_data.xlsx"
        ]
        
        df = None
        loaded_path = None
        
        for path in possible_paths:
            try:
                print(f"尝试加载聚类数据: {path}")
                df = pd.read_excel(path)
                loaded_path = path
                print(f"成功加载聚类数据: {path}, 数据形状: {df.shape}")
                break
            except FileNotFoundError:
                print(f"文件不存在: {path}")
                continue
            except Exception as e:
                print(f"加载 {path} 时出错: {e}")
                continue
        
        if df is None:
            print("所有路径都无法加载聚类数据文件")
            return pd.DataFrame()
        
        # 打印数据信息用于调试
        print(f"聚类数据列名: {list(df.columns)}")
        print(f"聚类数据前5行:\n{df.head()}")
        
        # 确保数据类型正确
        if '年份' in df.columns:
            df['年份'] = pd.to_numeric(df['年份'], errors='coerce').astype('Int64')
        
        # 聚类编号保持为字符串类型，不进行数字转换
        if '聚类编号' in df.columns:
            df['聚类编号'] = df['聚类编号'].astype(str)
            
        if '公司数量' in df.columns:
            df['公司数量'] = pd.to_numeric(df['公司数量'], errors='coerce').astype('Int64')
            
        print(f"处理后的聚类数据形状: {df.shape}")
        return df
        
    except Exception as e:
        print(f"加载聚类数据时发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
def load_cosine_matrix():
    """加载余弦相似度矩阵数据"""
    try:
        possible_paths = [
            "cosine_matrix.xlsx",
            "./cosine_matrix.xlsx", 
            "data/cosine_matrix.xlsx"
        ]
        
        df = None
        loaded_path = None
        
        for path in possible_paths:
            try:
                print(f"尝试加载余弦相似度矩阵: {path}")
                df = pd.read_excel(path, index_col=0)  # 第一列作为索引
                loaded_path = path
                print(f"成功加载余弦相似度矩阵: {path}, 数据形状: {df.shape}")
                break
            except FileNotFoundError:
                print(f"文件不存在: {path}")
                continue
            except Exception as e:
                print(f"加载 {path} 时出错: {e}")
                continue
        
        if df is None:
            print("所有路径都无法加载余弦相似度矩阵文件")
            return pd.DataFrame()
        
        print(f"余弦相似度矩阵索引前5个: {list(df.index[:5])}")
        print(f"余弦相似度矩阵列名前5个: {list(df.columns[:5])}")
        
        return df
        
    except Exception as e:
        print(f"加载余弦相似度矩阵时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_industry_similarity(cosine_df, target_company, cluster_companies):
    """
    计算目标公司与同行业公司的余弦相似度排序
    
    Args:
        cosine_df: 余弦相似度矩阵DataFrame
        target_company: 目标公司名称
        cluster_companies: 同行业公司列表（从聚类结果获得）
    
    Returns:
        包含相似度排序的DataFrame
    """
    if cosine_df.empty:
        print("余弦相似度矩阵为空")
        return pd.DataFrame()
    
    print(f"计算 {target_company} 与同行业公司的相似度")
    print(f"同行业公司列表: {cluster_companies[:5]}...")  # 显示前5个用于调试
    
    # 在矩阵中查找目标公司
    target_key = None
    for idx in cosine_df.index:
        if target_company in str(idx):
            target_key = idx
            print(f"找到目标公司索引键: {target_key}")
            break
    
    if target_key is None:
        print(f"在余弦相似度矩阵中未找到目标公司: {target_company}")
        # 尝试更宽松的匹配
        print("尝试更宽松的匹配...")
        for idx in cosine_df.index:
            if any(word in str(idx) for word in target_company.split()):
                target_key = idx
                print(f"通过宽松匹配找到目标公司: {target_key}")
                break
        
        if target_key is None:
            return pd.DataFrame()
    
    # 收集同行业公司的相似度数据
    similarity_data = []
    
    for company in cluster_companies:
        company = company.strip()
        # 关键修改：跳过目标公司本身
        if not company or company == target_company:
            print(f"跳过目标公司本身: {company}")
            continue
            
        # 在矩阵列中查找当前公司
        company_key = None
        for col in cosine_df.columns:
            if company in str(col):
                company_key = col
                break
        
        if company_key is None:
            # 尝试更宽松的匹配
            for col in cosine_df.columns:
                if any(word in str(col) for word in company.split() if len(word) > 1):
                    company_key = col
                    break
        
        if company_key is None:
            print(f"在余弦相似度矩阵中未找到公司: {company}")
            continue
        
        # 额外的安全检查：确保company_key不是目标公司本身的另一种表示
        if target_company in str(company_key):
            print(f"跳过目标公司的另一种表示: {company_key}")
            continue
        
        try:
            # 获取目标公司与当前公司的相似度值
            similarity_value = cosine_df.loc[target_key, company_key]
            
            print(f"公司 {company} (键: {company_key}) 与目标公司的相似度: {similarity_value}")
            
            # 从键中提取股票代码和公司名称 (格式: 股票代码_公司名称)
            if '_' in str(company_key):
                parts = str(company_key).split('_', 1)
                stock_code = parts[0]
                company_name = parts[1]
            else:
                stock_code = "未知"
                company_name = str(company_key)
            
            # 确保相似度值是数值类型且不是NaN
            if pd.isna(similarity_value):
                print(f"公司 {company} 的相似度值为NaN，跳过")
                continue
                
            similarity_data.append({
                '股票代码': stock_code,
                '公司名称': company_name,
                '余弦相似度': float(similarity_value)
            })
            
        except KeyError as e:
            print(f"矩阵中不存在键组合 [{target_key}, {company_key}]: {e}")
            continue
        except Exception as e:
            print(f"处理公司 {company} 时出错: {e}")
            continue
    
    if not similarity_data:
        print("未找到任何有效的相似度数据")
        return pd.DataFrame()
    
    # 创建DataFrame并按相似度排序
    result_df = pd.DataFrame(similarity_data)
    result_df = result_df.sort_values('余弦相似度', ascending=False)
    result_df = result_df.reset_index(drop=True)
    
    print(f"成功计算出 {len(result_df)} 个公司的相似度（已排除目标公司本身）")
    print(f"相似度值范围: {result_df['余弦相似度'].min():.4f} - {result_df['余弦相似度'].max():.4f}")
    
    return result_df


def find_company_cluster(cluster_df, company_name, year=None):
    """
    在聚类数据中查找指定公司所属的聚类信息
    
    Args:
        cluster_df: 聚类数据DataFrame
        company_name: 公司名称
        year: 年份（可选，如果不指定则返回所有年份的聚类信息）
    
    Returns:
        包含聚类信息的DataFrame
    """
    if cluster_df.empty:
        print("聚类数据为空，无法查找公司信息")
        return pd.DataFrame()
    
    print(f"在聚类数据中查找公司: {company_name}")
    print(f"聚类数据形状: {cluster_df.shape}")
    
    # 查找包含该公司的聚类记录
    matching_clusters = []
    
    for idx, row in cluster_df.iterrows():
        try:
            company_list = str(row.get('公司列表', ''))
            
            # 检查公司列表中是否包含目标公司（支持模糊匹配）
            if company_name in company_list or any(name.strip() == company_name for name in company_list.split(',') if name.strip()):
                # 如果指定了年份，则只返回匹配年份的记录
                if year is None or row.get('年份') == year:
                    print(f"找到匹配记录: 年份={row.get('年份')}, 聚类编号={row.get('聚类编号')}")
                    matching_clusters.append(row.to_dict())
        except Exception as e:
            print(f"处理聚类记录 {idx} 时出错: {e}")
            continue
    
    if matching_clusters:
        result_df = pd.DataFrame(matching_clusters)
        print(f"找到 {len(matching_clusters)} 条匹配的聚类记录")
        return result_df
    else:
        print(f"未找到公司 {company_name} 的聚类记录")
        return pd.DataFrame()

def load_and_process_data_for_viz(df, target_company_name):
    """
    将数据处理为可视化所需的格式，只处理目标公司的数据
    """
    processed_data = []
    
    # 先筛选出目标公司的数据
    company_data = df[df['公司名称'] == target_company_name]
    
    if company_data.empty:
        print(f"未找到公司 {target_company_name} 的数据")
        return pd.DataFrame()
    
    for _, row in company_data.iterrows():
        company_name = row['公司名称']
        stock_code = row['股票代码']
        year = row.get('年份', 2023)  # 使用get方法避免KeyError
        
        # 提取主营产品和词频
        for i in range(1, 6):  # 主营产品1到5
            product_col = f'主营产品{i}'
            freq_col = f'词频{i}'
            
            if product_col in row and freq_col in row:
                product = row[product_col]
                frequency = row[freq_col]
                
                # 只添加非空的产品数据
                if pd.notna(product) and pd.notna(frequency) and frequency > 0:
                    processed_data.append({
                        '公司名称': company_name,
                        '股票代码': stock_code,
                        '年份': year,
                        '主营产品': str(product).strip(),
                        '词频': int(frequency)
                    })
    
    result_df = pd.DataFrame(processed_data)
    print(f"处理后的数据行数: {len(result_df)}")
    return result_df

def create_plotly_charts(processed_df, company_name):
    """
    为单个公司创建Plotly图表
    """
    if processed_df.empty:
        print(f"没有可用于创建图表的数据: {company_name}")
        return None, None
    
    # 获取股票代码
    stock_code = processed_df['股票代码'].iloc[0]
    print(f"为公司 {company_name} ({stock_code}) 创建图表，数据行数: {len(processed_df)}")
    
    # 检查是否有多年数据
    unique_years = processed_df['年份'].unique()
    print(f"年份数据: {unique_years}")
    
    # 如果只有一年数据，创建简单的条形图
    if len(unique_years) == 1:
        # 单年数据 - 创建简单条形图
        year_data = processed_df[processed_df['年份'] == unique_years[0]]
        
        # 1. 简单条形图
        fig1 = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        for i, (_, row) in enumerate(year_data.iterrows()):
            fig1.add_trace(
                go.Bar(
                    x=[row['主营产品']],
                    y=[row['词频']],
                    name=row['主营产品'],
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f'<b>{row["主营产品"]}</b><br>词频: %{{y}}<extra></extra>',
                    opacity=0.85,
                    showlegend=False
                )
            )
        
        fig1.update_layout(
            title=dict(
                text=f'{company_name} ({stock_code}) - 主营产品词频分布 ({unique_years[0]}年)',
                x=0.5,
                font=dict(size=14, family="Arial Black")
            ),
            xaxis_title="主营产品",
            yaxis_title="词频",
            height=400,
            width=550,
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(r=50, l=50, t=60, b=100)
        )
        
        # 2. 饼图替代热力图
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Pie(
                labels=year_data['主营产品'],
                values=year_data['词频'],
                hovertemplate='<b>%{label}</b><br>词频: %{value}<br>占比: %{percent}<extra></extra>',
                marker=dict(colors=colors[:len(year_data)])
            )
        )
        
        fig2.update_layout(
            title=dict(
                text=f'{company_name} ({stock_code}) - 主营产品词频占比 ({unique_years[0]}年)',
                x=0.5,
                font=dict(size=14, family="Arial Black")
            ),
            height=400,
            width=550,
            font=dict(size=10),
            margin=dict(r=50, l=50, t=60, b=50)
        )
        
    else:
        # 多年数据 - 创建原有的堆叠图和热力图
        pivot_data = processed_df.pivot_table(index='年份', columns='主营产品', values='词频', fill_value=0)
        
        # 1. 堆叠条形图
        fig1 = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        for i, product in enumerate(pivot_data.columns):
            fig1.add_trace(
                go.Bar(
                    x=[str(year) for year in pivot_data.index],
                    y=pivot_data[product],
                    name=product,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f'<b>{product}</b><br>年份: %{{x}}<br>词频: %{{y}}<extra></extra>',
                    opacity=0.85
                )
            )
        
        fig1.update_layout(
            title=dict(
                text=f'{company_name} ({stock_code}) - 主营产品词频分布',
                x=0.5,
                font=dict(size=14, family="Arial Black")
            ),
            xaxis_title="年份",
            yaxis_title="词频",
            barmode='stack',
            height=400,
            width=550,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(r=120, l=50, t=60, b=50)
        )
        
        # 2. 热力图
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Heatmap(
                z=pivot_data.T.values,
                x=[str(year) for year in pivot_data.index],
                y=pivot_data.columns,
                colorscale='RdYlBu_r',
                showscale=True,
                hovertemplate='年份: %{x}<br>产品: %{y}<br>词频: %{z}<extra></extra>',
                colorbar=dict(
                    title="词频",
                    titleside="right",
                    title_font=dict(size=12),
                    tickfont=dict(size=10)
                )
            )
        )
        
        fig2.update_layout(
            title=dict(
                text=f'{company_name} ({stock_code}) - 词频热力图',
                x=0.5,
                font=dict(size=14, family="Arial Black")
            ),
            xaxis_title="年份",
            yaxis_title="主营产品",
            height=400,
            width=550,
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(r=50, l=50, t=60, b=50)
        )
    
    return fig1, fig2

# 预加载所有数据
data_store = {
    "一级产品": load_data("一级产品"),
    "二级产品": load_data("二级产品"),
    "三级产品": load_data("三级产品")
}

# 加载聚类数据
cluster_data = load_cluster_data()
# 加载余弦相似度矩阵
cosine_matrix = load_cosine_matrix()

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text("stock_input", "请输入股票代码或名称:", ""),
        ui.input_action_button("search_btn", "查询"),
        ui.hr(),
        ui.div(
            ui.h5("快速操作："),
            ui.input_action_button("clear_search", "清空查询", class_="btn-outline-secondary btn-sm"),
            ui.br(), ui.br(),
            ui.input_checkbox("enable_fuzzy", "启用模糊搜索", True),
        )
    ),
    ui.div(
        # 上方表格区域 - 集成所有查询功能
        ui.div(
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.h4("公司主营产品", style="margin: 0; display: inline-block;"),
                        ui.div(
                            ui.navset_tab(
                                ui.nav_panel("一级产品", value="一级产品"),
                                ui.nav_panel("二级产品", value="二级产品"),
                                ui.nav_panel("三级产品", value="三级产品"),
                                id="tab_select"
                            ),
                            style="float: right;"
                        ),
                        style="overflow: hidden;"
                    )
                ),

                # 数据统计信息
                ui.div(
                    ui.div(
                        ui.output_text("combined_stats"),
                        style="padding: 10px; font-size: 12px; background-color: #e9ecef;"
                    )
                ),
                # 主数据表格
                ui.div(
                    ui.h5("所有公司主营产品数据："),
                    ui.output_data_frame("all_table"),
                    style="margin-bottom: 20px;"
                ),
                # 查询结果区域
                ui.div(
                    ui.h5("查询结果："),
                    ui.output_ui("search_result_ui"),
                    style="border-top: 1px solid #dee2e6; padding-top: 15px;"
                )
            ),
            style="margin-bottom: 20px; height: 600px; overflow-y: auto;"
        ),
        
        # 下方可视化区域
        ui.div(
            # 第一行：图表区域
            ui.div(
                ui.div(
                    ui.card(
                        ui.card_header("主营产品词频堆叠分布"),
                        ui.div(
                            ui.output_ui("stacked_chart"),
                            style="padding: 10px;"
                        )
                    ),
                    class_="col-6",
                    style="padding-right: 10px;"
                ),
                ui.div(
                    ui.card(
                        ui.card_header("主营产品词频热力图"),
                        ui.div(
                            ui.output_ui("heatmap_chart"),
                            style="padding: 10px;"
                        )
                    ),
                    class_="col-6",
                    style="padding-left: 10px;"
                ),
                class_="row",
                style="margin-bottom: 15px;"
            ),
            # 第二行：聚类分析和相似度排序
            ui.div(
                ui.div(
                    ui.card(
                        ui.card_header("公司聚类分析结果"),
                        ui.div(
                            ui.output_ui("cluster_analysis"),
                            style="padding: 10px;"
                        )
                    ),
                    class_="col-6",
                    style="padding-right: 10px;"
                ),
                ui.div(
                    ui.card(
                        ui.card_header("行业内相似度排序"),
                        ui.div(
                            ui.output_ui("similarity_ranking"),
                            style="padding: 10px;"
                        )
                    ),
                    class_="col-6",
                    style="padding-left: 10px;"
                ),
                class_="row",
                style="margin-bottom: 15px;"
            ),
            # 第三行：AI解释行业分类原因（新位置）
            ui.div(
                ui.card(
                    ui.card_header("AI解释行业分类原因"),
                    ui.div(
                        ui.output_ui("ai_industry_explanation"),
                        style="padding: 10px;"
                    )
                ),
                style="margin-bottom: 15px;"
            ),
            style="height: auto;"  # 改为auto以适应内容
        ),
        class_="container-fluid",
        style="padding: 15px; height: auto;"  # 改为auto以适应内容
    ),
    title="基于公司主营产品的中证500行业分析App"
)

def server(input, output, session):

    @reactive.Calc
    def current_df():
        """根据当前选中的标签页返回对应层级的数据"""
        selected = input.tab_select()
        df = data_store[selected]
        return df

    @output
    @render.data_frame
    def all_table():
        """显示当前层级的主数据表格"""
        df = current_df()
        display_rows = 50  # 固定显示50行
        limited_df = df.head(display_rows)
        return render.DataGrid(
            limited_df,
            width="100%",
            height="200px",
            summary=f"显示 {len(limited_df)} 行，共 {len(df)} 行数据 | 当前层级: {input.tab_select()}"
        )

    search_trigger = reactive.Value(0)

    @reactive.Calc
    def matched_stock():
        """在当前层级数据中查询匹配的股票"""
        search_trigger.get()  # 强制依赖搜索触发器
        query = input.stock_input().strip()
        
        if not query:
            return pd.DataFrame()
        
        # 获取当前层级的数据
        df = current_df()
        
        # 根据输入类型进行查询
        if query.isdigit():
            # 股票代码查询（精确匹配）
            query_code = query.zfill(6)
            result = df[df["股票代码"] == query_code]
        else:
            # 公司名称查询
            if input.enable_fuzzy():
                # 模糊搜索：包含关键词即可
                result = df[df["公司名称"].str.contains(query, na=False, case=False)]
            else:
                # 精确搜索：完全匹配
                result = df[df["公司名称"].str.lower() == query.lower()]
        
        return result  # 返回所有匹配结果，不限制数量

    @reactive.Calc
    def current_company_name():
        """获取当前查询的公司名称"""
        result_df = matched_stock()
        if not result_df.empty:
            return result_df['公司名称'].iloc[0]
        return None

    @output
    @render.ui
    def cluster_analysis():
        """渲染聚类分析结果"""
        company_name = current_company_name()
        
        if not company_name:
            return ui.div(
                ui.p("请输入公司名称或股票代码进行查询",
                     style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
            )
        
        # 检查聚类数据是否加载成功
        if cluster_data.empty:
            return ui.div(
                ui.div(
                    ui.h6("聚类数据状态", style="color: #856404; margin-bottom: 10px;"),
                    ui.p("聚类数据未能正确加载。可能的原因：", style="color: #856404; margin-bottom: 5px;"),
                    ui.tags.ul(
                        ui.tags.li("cluster_data.xlsx 文件不存在于当前目录"),
                        ui.tags.li("文件格式不正确或已损坏"),
                        ui.tags.li("文件权限问题"),
                        style="color: #856404; margin-left: 20px;"
                    ),
                    ui.p("请检查控制台输出以获取详细错误信息。", style="color: #856404; margin-top: 10px; font-style: italic;"),
                    style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7;"
                )
            )
        
        try:
            # 查找公司的聚类信息
            cluster_info = find_company_cluster(cluster_data, company_name)
            
            if cluster_info.empty:
                return ui.div(
                    ui.div(
                        ui.h6(f" {company_name} - 聚类分析", style="color: #721c24; margin-bottom: 10px;"),
                        ui.p(f"在聚类记录中未找到 '{company_name}' 的相关信息。",
                             style="color: #721c24; margin-bottom: 8px;"),
                        ui.p("可能的原因:", style="color: #721c24; margin-bottom: 5px; font-weight: bold;"),
                        ui.tags.ul(
                            ui.tags.li("公司名称在聚类数据中的拼写不同"),
                            ui.tags.li("该公司未包含在聚类分析中"),
                            ui.tags.li("公司列表字段格式问题"),
                            style="color: #721c24; margin-left: 20px;"
                        ),
                        style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                    )
                )
            
            # 创建聚类信息显示UI
            cluster_ui_elements = []
            
            # 添加成功找到的标题
            cluster_ui_elements.append(
                ui.div(
                    ui.h6(f" {company_name} - 聚类分析结果", 
                          style="color: #155724; margin-bottom: 10px; font-weight: bold;"),
                    style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; margin-bottom: 15px;"
                )
            )
            
            # 遍历所有匹配的聚类记录
            for idx, (_, row) in enumerate(cluster_info.iterrows()):
                year = row.get('年份', 'N/A')
                cluster_id = row.get('聚类编号', 'N/A')
                company_count = row.get('公司数量', 'N/A')
                company_list = str(row.get('公司列表', ''))
                
                # 处理公司列表，限制显示长度
                if len(company_list) > 300:
                    display_company_list = company_list[:300] + "..."
                    full_list_hint = ui.p(f"(完整列表共 {len(company_list)} 个字符，已截取前300个字符显示)", 
                                        style="font-size: 10px; color: #6c757d; margin-top: 5px; font-style: italic;")
                else:
                    display_company_list = company_list
                    full_list_hint = ui.p("")
                
                # 创建每个聚类记录的卡片
                cluster_card = ui.div(
                    ui.div(
                        ui.div(
                            ui.strong(f"年份: {year}"),
                            style="display: inline-block; margin-right: 20px; color: #0c5460;"
                        ),
                        ui.div(
                            ui.strong(f"聚类编号: {cluster_id}"),
                            style="display: inline-block; margin-right: 20px; color: #0c5460;"
                        ),
                        ui.div(
                            ui.strong(f"公司数量: {company_count}"),
                            style="display: inline-block; color: #0c5460;"
                        ),
                        style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #bee5eb;"
                    ),
                    ui.div(
                        ui.strong("同一聚类的公司列表:", style="color: #0c5460; display: block; margin-bottom: 5px;"),
                        ui.div(
                            display_company_list,
                            style="font-size: 12px; line-height: 1.4; color: #495057; background-color: #f8f9fa; padding: 8px; border-radius: 3px; word-break: break-all;"
                        ),
                        full_list_hint
                    ),
                    style="background-color: #d1ecf1; padding: 12px; border-radius: 5px; border: 1px solid #bee5eb; margin-bottom: 10px;"
                )
                
                cluster_ui_elements.append(cluster_card)
            
            return ui.div(*cluster_ui_elements)
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            return ui.div(
                ui.div(
                    ui.h6("聚类分析错误", style="color: #721c24; margin-bottom: 10px;"),
                    ui.p(f"处理聚类数据时发生错误: {str(e)}",
                         style="color: #721c24; margin-bottom: 10px;"),
                    ui.details(
                        ui.summary("点击查看详细错误信息", style="color: #721c24; cursor: pointer;"),
                        ui.pre(error_trace, style="font-size: 10px; color: #721c24; background-color: #fff; padding: 10px; border-radius: 3px; overflow-x: auto;")
                    ),
                    style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                )
            )
    

    @output
    @render.ui
    def stacked_chart():
        """渲染堆叠条形图或条形图"""
        company_name = current_company_name()
        
        if not company_name:
            return ui.div(
                ui.p("请输入公司名称或股票代码进行查询",
                     style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
            )
        
        try:
            # 获取当前层级数据并处理为可视化格式
            current_data = current_df()
            print(f"当前数据层级: {input.tab_select()}, 公司: {company_name}")
            
            # 处理数据
            viz_data = load_and_process_data_for_viz(current_data, company_name)
            
            if viz_data.empty:
                return ui.div(
                    
                    ui.p(f"暂无 {company_name} 的可视化数据",
                         style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
                )
            
            # 创建图表
            fig1, _ = create_plotly_charts(viz_data, company_name)
            
            if fig1 is None:
                return ui.div(
                    ui.p(f"无法为 {company_name} 创建图表",
                         style="text-align: center; color: #dc3545; padding: 50px; font-size: 16px;")
                )
            
            # 将Plotly图表转换为HTML
            chart_html = fig1.to_html(include_plotlyjs='cdn', div_id="stacked_chart_div")
            return HTML(chart_html)
            
        except Exception as e:
            print(f"创建堆叠图表时出错: {str(e)}")
            return ui.div(
                ui.p(f"图表创建失败: {str(e)}",
                     style="text-align: center; color: #dc3545; padding: 50px; font-size: 16px;")
            )

    @output
    @render.ui
    def heatmap_chart():
        """渲染热力图或饼图"""
        company_name = current_company_name()
        
        if not company_name:
            return ui.div(
                ui.p("请输入公司名称或股票代码进行查询",
                     style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
            )
        
        try:
            # 获取当前层级数据并处理为可视化格式
            current_data = current_df()
            
            # 处理数据
            viz_data = load_and_process_data_for_viz(current_data, company_name)
            
            if viz_data.empty:
                return ui.div(
                    ui.p(f"暂无 {company_name} 的可视化数据",
                         style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
                )
            
            # 创建图表
            _, fig2 = create_plotly_charts(viz_data, company_name)
            
            if fig2 is None:
                return ui.div(
                    ui.p(f"无法为 {company_name} 创建图表",
                         style="text-align: center; color: #dc3545; padding: 50px; font-size: 16px;")
                )
            
            # 将Plotly图表转换为HTML
            chart_html = fig2.to_html(include_plotlyjs='cdn', div_id="heatmap_chart_div")
            return HTML(chart_html)
            
        except Exception as e:
            print(f"创建热力图/饼图时出错: {str(e)}")
            return ui.div(
                ui.p(f"图表创建失败: {str(e)}",
                     style="text-align: center; color: #dc3545; padding: 50px; font-size: 16px;")
            )

    @output
    @render.ui
    def search_result_ui():
        """显示查询结果的UI"""
        search_trigger.get()  # 监听搜索触发
        input.tab_select()    # 监听标签页切换
        
        query = input.stock_input().strip()
        result_df = matched_stock()

        # 如果没有输入查询条件
        if not query:
            return ui.div(
                ui.p("请输入股票代码或公司名称进行查询",
                     style="color: #6c757d; font-style: italic; text-align: center; padding: 20px;")
            )
        
        # 如果查询无结果
        if result_df.empty:
            return ui.div(
                ui.div(
                    ui.p(f"在【{input.tab_select()}】层级中未找到匹配的股票：'{query}'",
                         style="color: #dc3545; font-weight: bold;"),
                    ui.p("建议：", style="margin-top: 10px; margin-bottom: 5px;"),
                    ui.tags.ul(
                        ui.tags.li("检查公司名称拼写"),
                        ui.tags.li("尝试切换到其他产品层级"),
                        ui.tags.li("启用模糊搜索功能"),
                        ui.tags.li("清除当前筛选条件")
                    ),
                    style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                )
            )
        
        # 如果有查询结果，显示结果信息和表格
        return ui.div(
            ui.div(
                ui.p(f" 在【{input.tab_select()}】层级中找到 {len(result_df)} 条匹配记录：",
                     style="color: #155724; font-weight: bold; margin-bottom: 10px;"),
                style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; margin-bottom: 10px;"
            ),
            # 查询结果表格
            ui.output_data_frame("search_result_table")
        )

    @output
    @render.data_frame
    def search_result_table():
        """渲染查询结果表格"""
        result_df = matched_stock()
        
        if result_df.empty:
            return render.DataGrid(
                pd.DataFrame({"提示": ["暂无查询结果"]}),
                width="100%",
                height="100px"
            )
        
        return render.DataGrid(
            result_df,
            width="100%",
            height="200px",
            summary=f"查询结果: {len(result_df)} 条记录 | 产品层级: {input.tab_select()}"
        )

    
    @output
    @render.text
    def combined_stats():
        """显示综合统计信息"""
        df = current_df()
        search_trigger.get()
        result_df = matched_stock()
        query = input.stock_input().strip()
        
        # 数据统计
        data_stats = f" 数据层级: {input.tab_select()} | 显示: {len(df)} 家公司"
        if '年份' in df.columns:
            data_stats += f" | 年份: {df['年份'].min()}-{df['年份'].max()}"
        
        # 查询统计
        if query:
            if result_df.empty:
                search_stats = f" | 查询'{query}': 0条结果"
            else:
                search_mode = "模糊" if input.enable_fuzzy() else "精确"
                search_stats = f" | 查询'{query}': {len(result_df)}条结果({search_mode}匹配)"
        else:
            search_stats = " |  暂无查询"
            
        return data_stats + search_stats

    @reactive.Effect
    @reactive.event(input.search_btn)
    def handle_search():
        """处理搜索按钮点击事件"""
        search_trigger.set(search_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.clear_search)
    def clear_search():
        """处理清空搜索按钮点击事件"""
        ui.update_text("stock_input", value="")
        search_trigger.set(search_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.tab_select)
    def handle_tab_change():
        """处理标签页切换事件，自动重新执行查询"""
        if input.stock_input().strip():  # 如果有查询条件，则重新执行查询
            search_trigger.set(search_trigger.get() + 1)

    excel_path = r'产品聚类结果.xlsx'

    # 读取文件
    excel_file = pd.ExcelFile(excel_path)

    # 获取对应工作表数据
    df1 = excel_file.parse('最终聚类概览')
    df2 = excel_file.parse('公司-聚类映射')

    # 根据聚类编号列进行左连接
    df_concat = pd.merge(df1, df2, on='聚类编号', how='inner')

    # 调整列顺序
    df_concat = df_concat[['公司名称', '聚类编号', '公司数量', '公司列表']]

    def extract_company_name(company_str):
        """
        从包含股票代码的公司字符串中提取纯公司名称。
        支持处理 "公司名称(股票代码)"、"公司名称 股票代码" 和 "股票代码_公司名称" 格式。
        """
        # 去除前后空白
        company_str = company_str.strip()
        # 处理类似 "股票代码_公司名称" 格式
        if '_' in company_str:
            return company_str.split('_')[-1].strip()
        # 处理类似 "公司名称(股票代码)" 格式
        if '(' in company_str and company_str.endswith(')'):
            return company_str.split('(')[0].strip()
        # 处理类似 "公司名称 股票代码" 格式
        parts = company_str.split()
        if len(parts) > 1:
            # 假设最后一部分是股票代码，取前面部分拼接
            return ' '.join(parts[:-1])
        return company_str

    @reactive.Calc
    def ai_explanation():
        identifier = input.stock_input().strip()
        if identifier:
            # 筛选包含输入的股票代码或公司名称的行
            condition = df_concat['公司名称'].str.contains(identifier)
            filtered_df = df_concat[condition]
            if not filtered_df.empty:
                result = ""
                explanation = ""
                for _, row in filtered_df.iterrows():
                    # 将公司列表字符串按分隔符分割成列表
                    company_list = str(row['公司列表']).split(',')
                    # 提取纯公司名称
                    company_list = [extract_company_name(company) for company in company_list]
                    # 移除要查询的公司
                    company_list = [company for company in company_list if identifier not in company]
                    new_company_count = max(0, int(row['公司数量']) - 1)  # 公司数量减 1，最小为 0
                    new_company_list_str = ', '.join(company_list)
                    result += f"公司数量: {new_company_count}\n公司列表: {new_company_list_str}\n"
                    # 构建请求通义千问的问题
                    question = f"请解释为什么公司 {identifier} 和公司列表 {new_company_list_str} 是同行业公司。"
                    try:
                        response = Application.call(
                            # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
                            api_key="sk-2d1d971450e441ea8d6f1526fc2d78c7",
                            app_id='c960bbc8866e46939cbc069094d919db',  # 替换为实际的应用 ID
                            prompt=question
                        )
                        if response.status_code == 200:
                            explanation += response.output.text + "\n"
                    except Exception as e:
                        explanation += f"获取解释时出错: {str(e)}\n"

                return result + "\n解释:\n" + explanation
            else:
                return "未找到匹配的公司信息。"
        return "请输入股票代码或公司名称。"

    @output
    @render.ui
    def ai_industry_explanation():
        explanation = ai_explanation()
        return ui.div(
            ui.h6("AI 解释行业分类原因", style="margin-bottom: 10px;"),
            ui.pre(explanation, style="white-space: pre-wrap; word-break: break-word; padding: 10px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;")
        )

    @output
    @render.text
    def result_output():
        identifier = input.identifier_input()
        if identifier:
            # 筛选包含输入的股票代码或公司名称的行
            condition = df_concat['公司名称'].str.contains(identifier)
            filtered_df = df_concat[condition]
            if not filtered_df.empty:
                result = ""
                explanation = ""
                for _, row in filtered_df.iterrows():
                    # 将公司列表字符串按分隔符分割成列表
                    company_list = str(row['公司列表']).split(',')
                    # 提取纯公司名称
                    company_list = [extract_company_name(company) for company in company_list]
                    # 移除要查询的公司
                    company_list = [company for company in company_list if identifier not in company]
                    new_company_count = max(0, int(row['公司数量']) - 1)  # 公司数量减 1，最小为 0
                    new_company_list_str = ', '.join(company_list)
                    result += f"公司数量: {new_company_count}\n公司列表: {new_company_list_str}\n"
                    # 构建请求通义千问的问题
                    question = f"请解释为什么公司 {identifier} 和公司列表 {new_company_list_str} 是同行业公司。"
                    try:
                        response = Application.call(
                            # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
                            api_key="sk-2d1d971450e441ea8d6f1526fc2d78c7",
                            app_id='c960bbc8866e46939cbc069094d919db',  # 替换为实际的应用 ID
                            prompt=question
                        )
                        if response.status_code == 200:
                            explanation += response.output.text + "\n"
                    except Exception as e:
                        explanation += f"获取解释时出错: {str(e)}\n"

                return result + "\n解释:\n" + explanation
            else:
                return "未找到匹配的公司信息。"
        return "请输入股票代码或公司名称。"
    
    @output
    @render.ui
    def similarity_ranking():
        """渲染余弦相似度排序结果"""
        company_name = current_company_name()
    
        if not company_name:
            return ui.div(
                ui.p("请输入公司名称或股票代码进行查询",
                    style="text-align: center; color: #6c757d; padding: 50px; font-size: 16px;")
            )
        
        # 检查余弦相似度矩阵是否加载成功
        if cosine_matrix.empty:
            return ui.div(
                ui.div(
                    ui.h6("余弦相似度数据状态", style="color: #856404; margin-bottom: 10px;"),
                    ui.p("余弦相似度矩阵未能正确加载。可能的原因：", style="color: #856404; margin-bottom: 5px;"),
                    ui.tags.ul(
                        ui.tags.li("cosine_matrix.xlsx 文件不存在于当前目录"),
                        ui.tags.li("文件格式不正确或已损坏"),
                        ui.tags.li("文件权限问题"),
                        style="color: #856404; margin-left: 20px;"
                    ),
                    style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7;"
                )
            )
        
        # 检查聚类数据是否可用
        if cluster_data.empty:
            return ui.div(
                ui.div(
                    ui.h6("需要聚类数据", style="color: #856404; margin-bottom: 10px;"),
                    ui.p("余弦相似度分析需要先获得聚类数据以确定同行业公司范围。", 
                        style="color: #856404;"),
                    style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7;"
                )
            )
        
        try:
            # 关键修改：只获取2024年的聚类信息
            cluster_info = find_company_cluster(cluster_data, company_name, year=2024)
            
            if cluster_info.empty:
                return ui.div(
                    ui.div(
                        ui.h6(f"{company_name} - 余弦相似度分析 (2024年)", 
                            style="color: #721c24; margin-bottom: 10px;"),
                        ui.p("无法进行相似度分析，因为未找到该公司在2024年的聚类信息。",
                            style="color: #721c24;"),
                        ui.p("请确保该公司存在于2024年的聚类数据中。",
                            style="color: #721c24; font-style: italic;"),
                        style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                    )
                )
            
            # 获取2024年的同行业公司列表
            cluster_2024 = cluster_info[cluster_info['年份'] == 2024]
            if cluster_2024.empty:
                return ui.div(
                    ui.div(
                        ui.h6(f" {company_name} - 余弦相似度分析", 
                            style="color: #721c24; margin-bottom: 10px;"),
                        ui.p("未找到该公司在2024年的聚类数据。",
                            style="color: #721c24;"),
                        style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                    )
                )
            
            first_cluster = cluster_2024.iloc[0]
            company_list_str = str(first_cluster.get('公司列表', ''))
            cluster_companies = [comp.strip() for comp in company_list_str.split(',') if comp.strip()]
            
            print(f"2024年同行业公司数量: {len(cluster_companies)}")
            
            # 计算相似度排序
            similarity_df = calculate_industry_similarity(cosine_matrix, company_name, cluster_companies)
            
            if similarity_df.empty:
                return ui.div(
                    ui.div(
                        ui.h6(f" {company_name} - 余弦相似度分析 (2024年)", 
                            style="color: #721c24; margin-bottom: 10px;"),
                        ui.p("无法计算相似度，可能的原因：", style="color: #721c24; margin-bottom: 5px;"),
                        ui.tags.ul(
                            ui.tags.li("目标公司不在余弦相似度矩阵中"),
                            ui.tags.li("同行业公司在矩阵中的匹配度较低"),
                            ui.tags.li("数据格式不兼容"),
                            style="color: #721c24; margin-left: 20px;"
                        ),
                        style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                    )
                )
            
            # 创建成功的结果显示UI
            result_elements = []
            
            # 标题信息 - 明确标注2024年
            result_elements.append(
                ui.div(
                    ui.h6(f" {company_name} - 行业内相似度排序 (2024年聚类)", 
                        style="color: #155724; margin-bottom: 5px; font-weight: bold;"),
                    ui.p(f"基于2024年聚类结果分析了 {len(similarity_df)} 家同行业公司的相似度",
                        style="color: #155724; margin-bottom: 0; font-size: 12px;"),
                    style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; margin-bottom: 10px;"
                )
            )
            
            # 相似度表格 - 显示真实的相似度值
            table_rows = []
            for idx, row in similarity_df.head(10).iterrows():  # 只显示前10名
                rank = idx + 1
                stock_code = row['股票代码']
                company = row['公司名称']
                similarity = row['余弦相似度']
                
                # 根据相似度值设置颜色 - 调整阈值使其更合理
                if similarity >= 0.9:
                    color_class = "text-success"  # 绿色
                    bg_color = "background-color: #d4edda;"
                elif similarity >= 0.7:
                    color_class = "text-info"     # 蓝色
                    bg_color = "background-color: #d1ecf1;"
                elif similarity >= 0.5:
                    color_class = "text-warning"  # 黄色
                    bg_color = "background-color: #fff3cd;"
                else:
                    color_class = "text-muted"    # 灰色
                    bg_color = ""
                
                table_rows.append(
                    ui.tags.tr(
                        ui.tags.td(str(rank), style="text-align: center; font-weight: bold;"),
                        ui.tags.td(stock_code, style="text-align: center; font-family: monospace;"),
                        ui.tags.td(company, style="text-align: left;"),
                        ui.tags.td(f"{similarity:.6f}",  # 显示更多小数位以看清差异
                                style=f"text-align: center; font-weight: bold; {bg_color}",
                                class_=color_class),
                        style="border-bottom: 1px solid #dee2e6;"
                    )
                )
            
            similarity_table = ui.div(
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("排名", style="text-align: center; background-color: #f8f9fa;"),
                            ui.tags.th("股票代码", style="text-align: center; background-color: #f8f9fa;"),
                            ui.tags.th("公司名称", style="text-align: left; background-color: #f8f9fa;"),
                            ui.tags.th("余弦相似度", style="text-align: center; background-color: #f8f9fa;"),
                            style="border-bottom: 2px solid #dee2e6;"
                        )
                    ),
                    ui.tags.tbody(*table_rows),
                    class_="table table-sm",
                    style="margin-bottom: 0; font-size: 12px;"
                ),
                style="background-color: white; border-radius: 5px; overflow: hidden; border: 1px solid #dee2e6;"
            )
            
            result_elements.append(similarity_table)
            
            # 添加统计信息
            if len(similarity_df) > 0:
                min_sim = similarity_df['余弦相似度'].min()
                max_sim = similarity_df['余弦相似度'].max()
                avg_sim = similarity_df['余弦相似度'].mean()
                
                stats_info = ui.div(
                    ui.p(f"相似度统计：最高 {max_sim:.6f}，最低 {min_sim:.6f}，平均 {avg_sim:.6f}",
                        style="color: #6c757d; font-size: 11px; margin: 8px 0 0 0; text-align: center; font-style: italic;")
                )
                result_elements.append(stats_info)
            
            # 显示条数说明
            if len(similarity_df) > 10:
                result_elements.append(
                    ui.div(
                        ui.p(f"注：仅显示前10名，共有 {len(similarity_df)} 家同行业公司",
                            style="color: #6c757d; font-size: 11px; margin: 5px 0 0 0; text-align: center; font-style: italic;")
                    )
                )
            
            return ui.div(*result_elements)
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            return ui.div(
                ui.div(
                    ui.h6(" 相似度分析错误", style="color: #721c24; margin-bottom: 10px;"),
                    ui.p(f"处理相似度数据时发生错误: {str(e)}",
                        style="color: #721c24; margin-bottom: 10px;"),
                    ui.details(
                        ui.summary("点击查看详细错误信息", style="color: #721c24; cursor: pointer;"),
                        ui.pre(error_trace, style="font-size: 10px; color: #721c24; background-color: #fff; padding: 10px; border-radius: 3px; overflow-x: auto;")
                    ),
                    style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;"
                )
            )

app = App(app_ui, server)

# 运行应用
if __name__ == "__main__":
    app.run()