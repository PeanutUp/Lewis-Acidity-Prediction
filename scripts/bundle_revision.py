"""Export the checked manuscripts and a reproducible source/model archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    verification = json.loads((ROOT / 'reports/archive_verification.json').read_text())
    assert len(verification['models']) == 23
    exports = []
    for name in ['Digital_Discovery', 'JCIM', 'Supporting_Information', 'Revision_notes_ZH']:
        for suffix in ['.docx', '.pdf']:
            target = output / (name + suffix)
            shutil.copy2(ROOT / 'manuscript/review' / target.name, target)
            exports.append(target)
    for source, name in [('reports/comment_response_matrix.csv', 'comment_response_matrix.csv'),
                         ('reports/neural/seed_summary.csv', 'neural_seed_summary.csv'),
                         ('reports/refinement/metrics.csv', 'quantum_refinement_metrics.csv')]:
        target = output / name
        shutil.copy2(ROOT / source, target)
        exports.append(target)
    readme = output / 'README_ZH.txt'
    readme.write_text(f'''Lewis acidity 修订交付 · 2026-11-13

代码：https://github.com/PeanutUp/Lewis-Acidity-Prediction/pull/1
分支：codex/reliable-fia-benchmark
本次代码提交：{revision}
主要工作目录：/data2/hsm/Lewis-Acidity-Prediction

先阅读 Revision_notes_ZH.pdf：含18条PDF批注及逐条处理。
Digital_Discovery 和 JCIM 为同一研究内容的两份英文初投稿审阅稿；
Supporting_Information 为补充材料。Word可编辑，PDF已检查排版。
提供TeX源码和RSC官方模板；本机未编译TeX，审阅件以Word/PDF为准。

已完成23次神经模型训练。官方主测试6732分子，几何模型三种子MAE：
DimeNet++ 11.21±0.27；EGNN 12.08±0.16 kJ/mol。
种子平均与模型集成分开报告，不能混用。
量子校正主测试MAE 3.29，配体骨架组合留出3.35 kJ/mol；
此任务需要低级别酸与氟加合物FIA，输入成本高于只提供未配位酸的模型。
数值针对DSD-BLYP计算标签，不代表绝对实验精度。

修订替换了旧稿未经验证的精度和物理解释。原DimeNet保存配置重建出
旧2000分子中1813训练、105验证、82测试；此结论含明确重建假设，
EGNN历史训练清单仍缺失。新版实验采用独立划分并保存完整身份。

ZIP含代码、文稿、图、配置、训练日志、全部新预测及模型存储分块。
解压后先运行 python scripts/restore_artifacts.py，恢复73个压缩文件；
再按README安装Python3.10环境并准备公开数据。原始FIA49k几何与目标
未重复打入ZIP，可按版本化Figshare来源下载并校验。ZIP内附逐文件SHA256。
20项回归测试通过；23个权重与训练原件一致，分块恢复及3类模型预测往返已验证。

旧的独立修订目录已融合并删除。原始批注PDF保留在用户附件及仓库work中，
没有把私人批注原件上传公开仓库。

投稿前仍需作者确认署名、作者分工、资助及利益冲突，并取得新增归档的持久DOI。
AI辅助已披露。当前是待作者审核的研究稿，PR保持draft，未合并master或提交期刊。
''')
    exports.append(readme)
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--modified', '--others',
                                   '--exclude-standard', '-z'], cwd=ROOT).decode()
    paths = sorted({p for p in raw.split('\0') if p and (ROOT / p).is_file()})
    # Legacy binary training data and weights are accessible in the original Git
    # history; the archive contains the new frozen artifacts and source paths.
    allowed = {'.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.yaml', '.yml',
               '.toml', '.ini', '.sh', '.tex', '.svg', '.pdf', '.png', '.docx', '.xlsx'}
    selected = [p for p in paths if (ROOT / p).suffix in allowed
                or '.part-' in p or Path(p).name in {'.gitignore', 'LICENSE', 'Makefile'}]
    assert 'manuscript/rsc_official_article_template.tex' in selected
    manifest = {p: dict(bytes=(ROOT / p).stat().st_size, sha256=digest(ROOT / p))
                for p in selected}
    archive = output / 'Lewis_Acidity_Revision_2026-11-13.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in selected:
            z.write(ROOT / p, p)
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(dict(revision=revision, files=manifest),
                                                      ensure_ascii=False, indent=2) + '\n')
        z.write(readme, 'README_ZH.txt')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p, record in manifest.items():
            assert hashlib.sha256(z.read(p)).hexdigest() == record['sha256']
    exports.append(archive)
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in exports))
    print(f'Exported {len(exports)} deliverables; archive {archive.stat().st_size:,} bytes; {len(manifest)} verified entries')


if __name__ == '__main__':
    main()
