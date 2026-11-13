# MultiSpanQA data provenance

**Official dataset:** <https://github.com/haonan-li/MultiSpanQA>  
**Dataset paper:** Li et al., NAACL 2022, <https://aclanthology.org/2022.naacl-main.90/>  
**Dataset homepage/license notice:** <https://multi-span.github.io/> (CC BY-SA 4.0).

## Original input (to be verified on the student's machine)

- Release file / path: **TO VERIFY** (likely a gold-labelled split; do not assert `valid.json` until checking)
- Split: **TO VERIFY**
- SHA-256 of original file: **TO VERIFY**
- Upstream commit/tag or download date: **TO VERIFY**

The local preparation script normally writes a manifest next to the derived JSONL. Locate it:

```powershell
Get-ChildItem 'C:\Users\Lenovo\chain-of-verification\data' -Filter '*manifest.json' -Recurse
Get-Content 'C:\Users\Lenovo\chain-of-verification\data\multispanqa_pilot_60.manifest.json'
```

If that manifest exists, copy the *source_file*, *source_sha256*, and selection fields into `data/provenance_template.json`, replacing the absolute private Windows path with the public upstream-relative path. If absent, compute the checksum against the **exact original JSON**:

```powershell
Get-FileHash 'C:\path\to\actual\original\valid.json' -Algorithm SHA256
```

## Derived subset (verified from the archived file)

- 60 selected question IDs, 170 reference spans.
- Selection: at least two unique spans, each at most three context tokens; seed 42; limit 60.
- BIO labels extracted from contexts for gold creation; the context never goes to the language model.
- Exact CoVe paper 418-question selection not recovered.
- Derived JSONL SHA-256: `d8431c39be7d3988da308a278ab12a1c4b38bce0dd90289a4ec6c04f96cc9e66`.

### Licensing note

Keep attribution to MultiSpanQA authors. The upstream dataset site says CC BY-SA 4.0; redistribution of this derived labelled subset should comply with that license and applicable attribution/share-alike conditions. This does not automatically grant a license to separately sourced third-party CoVe implementation code.
