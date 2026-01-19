# Notion Database System – Conceptual & Relational Model

> **Software Chosen:** Notion
>
> **Author:** Shaik Chishti
> **Title:** DBMS Hand-in Assignment<br>
> **Sources:** Official Notion Developer & Help Documentation + Personal Analysis Notes

---

## 1. Brief Overview of Notion

Notion is a productivity and knowledge-management tool used to:

* Plan and create structured content
* Build highly customizable data structures
* Replace multiple applications with a single unified workspace

At its core, **everything in Notion is represented as blocks**. A block may:

* Exist independently
* Contain child blocks
* Be inherited or nested
* Act as both content and data

This block-first design is the foundation for Notion’s database system.

---

## 2. High-Level Architecture

Conceptually, Notion’s architecture revolves around six major entities:

1. Databases
2. Pages
3. Blocks
4. Database Properties
5. Properties & Values
6. Relations

Each of these entities is stored independently and connected through explicit relationships, enabling scalable and flexible database behavior.

---

## 3. Core Entities and Their Roles

### 3.1 Databases

A **Database** defines structure and schema. It does not directly store rows; instead, it owns pages.

**Key characteristics:**

* Can have multiple pages
* Can have multiple database properties
* Can exist inline or as a full page

**Fields (as per official API & notes):**

* `object` (string)
* `id` (UUIDv4)
* `data_sources` (array of objects)
* `created_time` (string)
* `created_by` (user)
* `last_edited_time` (string)
* `title` (array of rich-text objects)
* `description` (array of rich-text objects)
* `icon` (object)
* `cover` (object)
* `parent` (object)
* `url` (string)
* `archived` (boolean)
* `in_trash` (boolean)
* `is_inline` (boolean)
* `public_url` (string)
* `page_id` (UUIDv4)

---

### 3.2 Pages

A **Page** represents a row inside a database. Every row in a Notion database is a page.

**Key characteristics:**

* Belongs to exactly one database
* Contains blocks as content
* Stores property values
* Participates in relations

**Fields:**

* `object` (string)
* `id` (UUIDv4)
* `created_by` (user)
* `last_edited_time` (string)
* `last_edited_by` (user)
* `archived` (boolean)
* `in_trash` (boolean)
* `icon` (object)
* `cover` (object)
* `properties` (object)
* `parent` (object)
* `url` (string)
* `public_url` (string)

---

### 3.3 Blocks

Blocks are the **atomic units of Notion**. Everything — pages, headings, lists, database rows — is ultimately a block.

Blocks support hierarchical nesting via parent-child relationships.

**Fields:**

* `object` (string)
* `id` (UUIDv4)
* `parent` (object)
* `type` (string enum)
* `created_time` (ISO 8601 datetime)
* `last_edited_time` (ISO 8601 datetime)
* `last_edited_by` (user)
* `in_trash` (boolean)
* `archived` (boolean)
* `has_children` (boolean)
* `properties` (object)
* `icon` (object)
* `cover` (object)
* `url` (string)
* `public_url` (string)

Blocks can have **multiple sub-blocks**, enabling deep nesting.

---

### 3.4 Database Properties

**Database Properties** define the schema of a database.

Examples include:

* Title
* Number
* Select / Multi-select
* Relation
* Rollup
* Formula

**Fields:**

* `id` (string)
* `name` (string)
* `description` (string)
* `type` (string enum)

A database can have **multiple database properties**.

---

### 3.5 Properties & Values

**Properties & Values** represent actual data stored in pages.

Each property value:

* Belongs to one page
* References one database property
* Stores typed JSON data

**Fields:**

* `id` (string)
* `name` (string)
* `description` (string)
* `type` (string enum)
* `created_by` (user)
* `creation_properties` (JSON)

Each page can have **multiple property values**, and each value may connect to blocks.

---

### 3.6 Relations

Relations define **many-to-many links** between pages.

They are graph edges rather than traditional foreign keys.

**Fields:**

* `id` (string)
* `is_parent` (boolean)
* `is_child` (boolean)
* `has_child` (boolean)
* `child_properties` (JSON)
* `relation_type` (string enum)

Pages can have **multiple relations**, including self-relations.

---

## 4. Relationships Between Entities (Logical View)

* A **Database** has many **Pages**
* A **Database** has many **Database Properties**
* A **Page** has many **Property Values**
* A **Page** has many **Blocks**
* **Relations** connect Pages ↔ Pages
* **Property Values** connect Pages ↔ Database Properties

This structure enables schema flexibility, graph traversal, and real-time computation.

---

## 5. Key Design Insights

* Everything is block-based
* Rows are pages, not tuples
* Schema is metadata-driven
* Relations form a graph
* Rollups and formulas are computed, not stored
* Deep nesting enables content + data unification

This design allows Notion databases to scale while remaining flexible and extensible.

---

## 6. References (Official Documentation)

### Notion Developer Documentation

* [https://developers.notion.com](https://developers.notion.com)
* [https://developers.notion.com/reference](https://developers.notion.com/reference)

### Notion Help Center

* [https://www.notion.com/help/database-properties](https://www.notion.com/help/database-properties)
* [https://www.notion.com/help/relations-and-rollups](https://www.notion.com/help/relations-and-rollups)

### Personal Documentation

* [https://github.com/chishtil1730/IML-LAB/tree/main/notion_doc](https://github.com/chishtil1730/IML-LAB/tree/main/notion_doc)

---

*This document consolidates official documentation with personal system-level analysis to describe how Notion internally models its database architecture.*


**_!This document has the text extracted from my hand notes_**