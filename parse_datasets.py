import re
import json

def parse_markdown_to_json(md_file_path, output_json_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    datasets = []
    skipped = []
    
    # Split by numbered entries
    entries = re.split(r'\n(?=\d+\.)', content.strip())
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Skip commented lines (markdown comments or HTML comments)
        if re.match(r'^<!--', entry) or re.match(r'^\[//\]:', entry):
            continue
        
        # Extract entry number first to track it
        entry_num_match = re.match(r'^(\d+)\.', entry)
        entry_num = entry_num_match.group(1) if entry_num_match else "unknown"
        
        # Parse the entry structure with semicolon separator
        # Format: Number. Authors; Title, GSE, **files** info platform
        match = re.match(r'^(\d+)\.\s+(.+?);\s+(.+?),\s*(GSE\d+)\s*,?\s*(.+)$', entry, re.DOTALL)
        
        if not match:
            print(f"⚠️  Warning: Could not parse entry {entry_num}: {entry[:80]}...")
            skipped.append(entry_num)
            continue
        
        entry_num, authors_str, title, gse_accession, rest = match.groups()
        
        # Clean up authors and title
        authors = authors_str.strip()
        title = title.strip()
        
        # Remove GSE accession from title if it was captured there
        title = re.sub(r',?\s*GSE\d+\s*,?', '', title).strip()
        
        # Extract all data files with organism markers
        file_pattern = r'\*\*([^\*]+)\*\*\s*\(([HM]m)\)'
        file_matches = list(re.finditer(file_pattern, rest))
        
        if not file_matches:
            print(f"⚠️  Warning: No files found in entry {entry_num}")
            skipped.append(entry_num)
            continue
        
        # For entries with multiple files, we need to parse each file's metadata separately
        # Each file may have its own source and platform
        for idx, file_match in enumerate(file_matches):
            data_file_name = file_match.group(1).strip()
            organism_code = file_match.group(2)
            organism = "Human" if organism_code == "Hm" else "Mouse"
            
            # Extract metadata for this specific file
            # Get text between this file and the next file (or end of string)
            start_pos = file_match.end()
            if idx + 1 < len(file_matches):
                end_pos = file_matches[idx + 1].start()
            else:
                end_pos = len(rest)
            
            file_metadata = rest[start_pos:end_pos].strip()
            
            # Remove leading comma if present
            file_metadata = re.sub(r'^,\s*', '', file_metadata)
            # Element AVITI, MGISEQ-2000RS, Illumina NovaSeq X
            # Enhanced platform pattern to catch more variations
            platform_pattern = r'(Illumina\s+NovaSeq\s+\d+|Illumina\s+HiSeq\s+X\s+Ten|Illumina\s+HiSeq\s+\d+|Illumina\s+NextSeq\s+\d+|Illumina\s+MiSeq|NextSeq\s+\d+|NovaSeq\s+\d+|HiSeq\s+X\s+Ten|HiSeq\s+\d+|MiSeq)'
            platform_match = re.search(platform_pattern, file_metadata, re.IGNORECASE)
            
            if platform_match:
                platform = platform_match.group(0).strip()
                source = file_metadata[:platform_match.start()].strip()
            else:
                # If no platform found, assign all to source
                platform = "Unknown Platform"
                source = file_metadata.strip()
            
            # Clean up source (remove trailing punctuation and whitespace)
            source = re.sub(r'[,\s]+$', '', source).strip()
            
            # If source is empty, mark as Unknown
            if not source:
                source = "Unknown"
            
            # Generate a unique ID from GSE and filename
            file_id_match = re.search(r'(GSM\d+)', data_file_name)
            if file_id_match:
                file_identifier = file_id_match.group(1).lower()
            else:
                # Try to extract other meaningful identifiers
                alt_match = re.search(r'^([^_]+)', data_file_name)
                if alt_match:
                    file_identifier = alt_match.group(1).lower()
                else:
                    file_identifier = f"file{idx+1}"
            
            dataset_id = f"{gse_accession.lower()}-{file_identifier}"
            # Clean ID: only keep alphanumeric and hyphens
            dataset_id = re.sub(r'[^a-z0-9-]', '', dataset_id)
            
            dataset = {
                "id": dataset_id,
                "authors": authors,
                "title": title,
                "gseAccession": gse_accession,
                "dataFileName": data_file_name,
                "organism": organism,
                "source": source,
                "platform": platform
            }
            
            datasets.append(dataset)
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully parsed {len(datasets)} datasets")
    print(f"📁 Output saved to: {output_json_path}")
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} entries: {', '.join(skipped)}")
    
    # Print detailed summary
    print("\n📊 Summary:")
    print(f"Total datasets: {len(datasets)}")
    
    organisms = {}
    platforms = {}
    sources = {}
    gse_counts = {}
    
    for d in datasets:
        organisms[d['organism']] = organisms.get(d['organism'], 0) + 1
        platforms[d['platform']] = platforms.get(d['platform'], 0) + 1
        sources[d['source']] = sources.get(d['source'], 0) + 1
        gse_counts[d['gseAccession']] = gse_counts.get(d['gseAccession'], 0) + 1
    
    print("\nBy Organism:")
    for org, count in sorted(organisms.items()):
        print(f"  {org}: {count}")
    
    print("\nBy Platform (top 10):")
    for plat, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {plat}: {count}")
    
    print(f"\nUnique GSE accessions: {len(gse_counts)}")
    print(f"Unique sources: {len(sources)}")
    
    return datasets

if __name__ == "__main__":
    # Usage
    md_file = "./scATAC25100-info.md"
    json_file = "./datasets.json"
    
    try:
        datasets = parse_markdown_to_json(md_file, json_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{md_file}'")
        print("Please make sure the markdown file exists at the specified path.")
    except Exception as e:
        print(f"❌ Error during parsing: {str(e)}")
        import traceback
        traceback.print_exc()